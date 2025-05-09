"""
Generate the nodes and edges file for the MIRA domain knowledge graph.

After these are generated, see the /docker folder in the repository for loading
a neo4j instance.

Example command for local bulk import on mac with neo4j 4.x:

.. code::

    neo4j-admin import --database=mira \
        --delimiter='TAB' \
        --force \
        --skip-duplicate-nodes=true \
        --skip-bad-relationships=true \
        --nodes ~/.data/mira/demo/import/nodes.tsv.gz \
        --relationships ~/.data/mira/demo/import/edges.tsv.gz

Then, restart the neo4j service with homebrew ``brew services neo4j restart``
"""

import csv
import gzip
import json
import pickle
import typing
from collections import Counter, defaultdict
from datetime import datetime
from operator import methodcaller
from pathlib import Path
from typing import Dict, NamedTuple, Sequence, Union, Optional

import biomappings
import bioontologies
import click
import pyobo
import pystow
import networkx
from bioontologies import obograph
from bioregistry import manager
from pydantic import BaseModel, Field
from pyobo.struct import part_of, is_a
from pyobo.sources import ontology_resolver
from pyobo.api.utils import get_version
from pyobo.utils.path import prefix_directory_join
from tabulate import tabulate
from tqdm.auto import tqdm
from typing_extensions import Literal

from mira.dkg.askemo import get_askemo_terms, get_askemosw_terms, get_askem_climate_ontology_terms
from mira.dkg.models import EntityType
from mira.dkg.resources import SLIMS, get_ncbitaxon
from mira.dkg.resources.extract_ncit import get_ncit_subset
from mira.dkg.resources.probonto import get_probonto_terms
from mira.dkg.units import get_unit_terms
from mira.dkg.physical_constants import get_physical_constant_terms
from mira.dkg.constants import EDGE_HEADER, NODE_HEADER
from mira.dkg.utils import PREFIXES
from mira.dkg.models import Synonym, Xref
from mira.dkg.resources.cso import get_cso_obo
from mira.dkg.resources.geonames import get_geonames_terms
from mira.dkg.resources.extract_eiffel_ontology import get_eiffel_ontology_terms
from mira.dkg.resources.uat import get_uat
from mira.dkg.generate_obo_graphs import download_convert_ncbitaxon_obo_to_graph

MODULE = pystow.module("mira")
DEMO_MODULE = MODULE.module("demo", "import")
EDGE_NAMES_PATH = DEMO_MODULE.join(name="relation_info.json")
METAREGISTRY_PATH = DEMO_MODULE.join(name="metaregistry.json")

OBSOLETE = {"oboinowl:ObsoleteClass", "oboinowl:ObsoleteProperty"}


class DKGConfig(BaseModel):
    use_case: str
    prefix: Optional[str] = None
    func: Optional[typing.Callable] = None
    iri: Optional[str] = None,
    prefixes: typing.List[str] = Field(default_factory=list)


cases: Dict[str, DKGConfig] = {
    "epi": DKGConfig(
        use_case="epi",
        prefix="askemo",
        func=get_askemo_terms,
        iri="https://github.com/gyorilab/mira/blob/main/mira/dkg/askemo/askemo.json",
        prefixes=PREFIXES,
    ),
    "space": DKGConfig(
        use_case="space",
        prefix="askemosw",
        func=get_askemosw_terms,
        iri="https://github.com/gyorilab/mira/blob/main/mira/dkg/askemo/askemosw.json",
    ),
    "eco": DKGConfig(
        use_case="eco",
        prefixes=["hgnc", "ncbitaxon", "ecocore", "probonto", "reactome"],
    ),
    "genereg": DKGConfig(
        use_case="genereg",
        prefixes=["hgnc", "go", "wikipathways", "probonto"],
    ),
    "climate": DKGConfig(
        use_case="climate",
        prefix="askem.climate",
        func=get_askem_climate_ontology_terms,
        prefixes=["probonto"],
        iri="https://github.com/gyorilab/mira/blob/main/mira/dkg/askemo/askem.climate.json",
    ),
}


class UseCasePaths:
    """A configuration containing the file paths for use case-specific files."""

    def __init__(self, use_case: str, config: Optional[DKGConfig] = None):
        self.use_case = use_case
        self.config = config or cases[self.use_case]
        self.askemo_prefix = self.config.prefix
        self.askemo_getter = self.config.func
        self.askemo_url = self.config.iri
        self.prefixes = self.config.prefixes

        self.module = MODULE.module(self.use_case)
        self.UNSTANDARDIZED_NODES_PATH = self.module.join(
            name="unstandardized_nodes.tsv"
        )
        self.UNSTANDARDIZED_EDGES_PATH = self.module.join(
            name="unstandardized_edges.tsv"
        )
        self.SUB_EDGE_COUNTER_PATH = self.module.join(
            name="count_subject_prefix_predicate.tsv"
        )
        self.SUB_EDGE_TARGET_COUNTER_PATH = self.module.join(
            name="count_subject_prefix_predicate_target_prefix.tsv"
        )
        self.EDGE_OBJ_COUNTER_PATH = self.module.join(
            name="count_predicate_object_prefix.tsv"
        )
        self.EDGE_COUNTER_PATH = self.module.join(name="count_predicate.tsv")
        self.NODES_PATH = self.module.join(name="nodes.tsv.gz")
        self.EDGES_PATH = self.module.join(name="edges.tsv.gz")
        self.EMBEDDINGS_PATH = self.module.join(name="embeddings.tsv.gz")

        prefixes = list(self.prefixes)
        if self.askemo_prefix:
            prefixes.append(self.askemo_prefix)
        if self.use_case == "space":
            prefixes.append("uat")
        if self.use_case == "climate":
            prefixes.append("eiffel")
        self.EDGES_PATHS: Dict[str, Path] = {
            prefix: self.module.join("sources", name=f"edges_{prefix}.tsv")
            for prefix in prefixes
        }
        self.RDF_TTL_PATH = self.module.join(name="dkg.ttl.gz")


LABELS = {
    "http://www.w3.org/2000/01/rdf-schema#isDefinedBy": "is defined by",
    "rdf:type": "type",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type": "type",
    # FIXME deal with these relations
    "http://purl.obolibrary.org/obo/uberon/core#proximally_connected_to": "proximally_connected_to",
    "http://purl.obolibrary.org/obo/uberon/core#extends_fibers_into": "proximally_connected_to",
    "http://purl.obolibrary.org/obo/uberon/core#channel_for": "proximally_connected_to",
    "http://purl.obolibrary.org/obo/uberon/core#distally_connected_to": "proximally_connected_to",
    "http://purl.obolibrary.org/obo/uberon/core#channels_into": "channels_into",
    "http://purl.obolibrary.org/obo/uberon/core#channels_from": "channels_from",
    "http://purl.obolibrary.org/obo/uberon/core#subdivision_of": "subdivision_of",
    "http://purl.obolibrary.org/obo/uberon/core#protects": "protects",
    "http://purl.obolibrary.org/obo/uberon/core#posteriorly_connected_to": "posteriorly_connected_to",
    "http://purl.obolibrary.org/obo/uberon/core#evolved_from": "evolved_from",
    "http://purl.obolibrary.org/obo/uberon/core#anteriorly_connected_to": "anteriorly_connected_to",
}

DEFAULT_VOCABS = [
    "oboinowl",
    "ro",
    "bfo",
    "owl",
    "rdfs",
    "bspo",
    # "gorel",
    "iao",
    # "sio",
    "omo",
    "debio",
]


class NodeInfo(NamedTuple):
    curie: str  # the id used in neo4j
    prefix: str  # the field used for neo4j labels. can contain semicolon-delimited
    label: str  # the human-readable label
    synonyms: str
    deprecated: Literal["true", "false"]  # need this for neo4j
    type: EntityType
    definition: str
    xrefs: str
    alts: str
    version: str
    property_predicates: str
    property_values: str
    xref_types: str
    synonym_types: str


def extract_nodes_edges_from_pyobo_terms(term_getter, resource_prefix):
    nodes, edges = [], []
    if resource_prefix in {"geonames"}:
        entity_type = "individual"
    elif resource_prefix in {"ncit", "ncbitaxon", "eiffel", "cso"}:
        entity_type = "class"
    for term in tqdm(term_getter(), unit="term"):
        if resource_prefix != "ncbitaxon":
            nodes.append(
                {
                    "id": term.curie,
                    "name": term.name,
                    "type": entity_type,
                    "description": term.definition if term.definition else "",
                    "obsolete": False if not term.is_obsolete else True,
                    "synonyms": [Synonym(value=syn.name,
                                         type=f"{syn.type.reference.prefix}:"
                                              f"{syn.type.reference.identifier}")
                                 for syn in term.synonyms],
                    "alts": term.alt_ids,
                    "xrefs": [Xref(id=_id, type=type) for _id, type in
                              zip(term.xrefs, term.xref_types)],
                    "properties": dict(term.properties),
                }
            )
        else:
            nodes.append(
                {
                    "id": term.curie,
                    "name": term.name,
                    "type": entity_type,
                    "description": term.definition if term.definition else "",
                    "obsolete": False if not term.is_obsolete else True,
                    "synonyms": [Synonym(value=syn.name,
                                         type=f"{syn.type.reference.prefix}:"
                                              f"{syn.type.reference.identifier}")
                                 for syn in term.synonyms],
                    "alts": [f"{reference.prefix}:{reference.identifier}" for
                             reference in term.alt_ids],
                    "xrefs": [Xref(id=f"{reference.prefix}:"
                                      f"{reference.identifier}", type="")
                              for reference in term.xrefs],
                    "properties": dict(term.properties),
                }
            )
        if resource_prefix != "eiffel":
            for parent in term.get_relationships(part_of):
                edges.append(
                    {
                        "source_curie": term.curie,
                        "target_curie": parent.curie,
                        "type": "part_of",
                        "pred": part_of.curie.lower(),
                        "source": resource_prefix,
                        "graph": resource_prefix,
                        "version": "",
                    }
                )
        else:
            for typedef, object_references in term.relationships.items():
                for object_reference in object_references:
                    edges.append(
                        {
                            "source_curie": term.curie,
                            "target_curie": object_reference.curie,
                            "type": typedef.name.replace(" ", "").lower(),
                            "pred": typedef.curie,
                            "source": resource_prefix,
                            "graph": resource_prefix,
                            "version": "",
                        }
                    )
    return nodes, edges


def extract_probonto_nodes_edges():
    probonto_nodes, probonto_edges = [], []
    for term in tqdm(get_probonto_terms(), unit="term"):
        curie, name, parameters = (
            term["curie"],
            term["name"],
            term["parameters"],
        )
        properties = {
            "has_parameter": [parameter["name"].replace("\n", " ") for parameter
                              in
                              parameters]
        }
        probonto_nodes.append(
            {
                "id": curie,
                "name": name,
                "type": "class",
                "description": "",
                "obsolete": False,
                "xrefs": [Xref(id=eq.get("curie", ""), type="askemo:0000016")
                          for eq in term.get("equivalent", [])],
                "properties": properties

            }
        )
        for parameter in term.get("parameters", []):
            parameter_curie, parameter_name = (
                parameter["curie"],
                parameter["name"],
            )
            synonyms = []
            synonym_types = []
            parameter_symbol = parameter.get("symbol")
            if parameter_symbol:
                synonyms.append(parameter_symbol)
                synonym_types.append("referenced_by_latex")
            parameter_short = parameter.get("short_name")
            if parameter_short:
                synonyms.append(parameter_short)
                synonym_types.append("oboInOwl:hasExactSynonym")
            synonyms_list = [Synonym(value=value, type=type) for value, type in
                             zip(synonyms, synonym_types)]
            probonto_nodes.append(
                {
                    "id": parameter_curie,
                    "name": parameter_name,
                    "type": "class",
                    "description": "",
                    "obsolete": False,
                    "synonyms": synonyms_list
                }
            )
            probonto_edges.append(
                {
                    "source_curie": curie,
                    "target_curie": parameter_curie,
                    "type": "has_parameter",
                    "pred": "probonto:c0000062",
                    "source": "probonto",
                    "graph": "https://raw.githubusercontent.com/probonto/ontologymaster/probonto4ols.owl",
                    "version": "2.5",
                }
            )
    return probonto_nodes, probonto_edges


def extract_wikidata_nodes_edges():
    wikidata_nodes, wikidata_edges = [], []
    for wikidata_id, label, description, synonyms, xrefs in tqdm(
        get_unit_terms(), unit="unit"):
        synonyms_list = [Synonym(value=value, type="") for value in synonyms]
        xrefs_list = [Xref(id=_id, type="oboinowl:hasDbXref") for _id in xrefs]
        wikidata_nodes.append(
            {
                "id": f"wikidata:{wikidata_id}",
                "name": label,
                "type": "class",
                "description": description,
                "synonyms": synonyms_list,
                "xrefs": xrefs_list,
                "obsolete": False
            }
        )

    for (wikidata_id, label, description, synonyms, xrefs, value, formula,
         symbols) in tqdm(get_physical_constant_terms()):
        synonym_types, synonym_values = [], []
        for syn in synonyms:
            synonym_values.append(syn)
            synonym_types.append("oboInOwl:hasExactSynonym")
        for symbol in symbols:
            synonym_values.append(symbol)
            synonym_types.append("debio:0000031")

        synonyms_list = [Synonym(value=value, type=type) for value, type
                         in zip(synonym_values, synonym_types)]
        xrefs_list = [Xref(id=_id, type="oboinowl:hasDbXref") for _id in xrefs]
        if value:
            properties = {"debio:0000042": [str(value)]}
        else:
            properties = {}
        wikidata_nodes.append(
            {
                "id": f"wikidata:{wikidata_id}",
                "name": label,
                "obsolete": False,
                "type": "class",
                "description": description,
                "synonyms": synonyms_list,
                "xrefs": xrefs_list,
                "properties": properties
            }
        )
    return wikidata_nodes, wikidata_edges


def add_resource_to_dkg(resource_prefix: str):
    if resource_prefix == "probonto":
        return extract_probonto_nodes_edges()
    elif resource_prefix == "geonames":
        return extract_nodes_edges_from_pyobo_terms(get_geonames_terms,
                                                       "geonames")
    elif resource_prefix == "ncit":
        return extract_nodes_edges_from_pyobo_terms(get_ncit_subset,
                                                       "ncit")
    elif resource_prefix == "ncbitaxon":
        return extract_nodes_edges_from_pyobo_terms(get_ncbitaxon,
                                                       "ncbitaxon")
    elif resource_prefix == "eiffel":
        return extract_nodes_edges_from_pyobo_terms(
            get_eiffel_ontology_terms, "eiffel")
    elif resource_prefix == "cso":
        return extract_nodes_edges_from_pyobo_terms(get_cso_obo,
                                                       "cso")
    elif resource_prefix == "wikidata":
        # combine retrieval of wikidata constants and units
        return extract_wikidata_nodes_edges()
    else:
        # handle resource names that we don't process
        return [], []

def extract_ontology_subtree(curie: str, add_subtree: bool = False):
    """Takes in a curie and extracts the information from the
    entry in its respective resource ontology to add as a node into the
    Epidemiology DKG.

    There is an option to extract all the information from the entries
    under the corresponding entry's subtree in its respective ontology.
    Relation information is also extracted with this option.

    Execution of this method will take a few seconds as the pickled
    graph object has to be loaded.

    Currently we only support the addition of ncbitaxon terms.

    Parameters
    ----------
    curie :
        The curie for the entry that will be added as a node to the
        Epidemiology DKG.
    add_subtree :
        Whether to add all the nodes and relations under the entry's subtree

    Returns
    -------
    nodes : List[dict]
        A list of node information added to the DKG, where each node is
        represented as a dictionary.
    edges : List[dict]
        A list of edge information added to the DKG, where each edge is
        represented as a dictionary.
    """
    nodes, edges = [], []
    resource_prefix = curie.split(":")[0]
    if resource_prefix == "ncbitaxon":
        type = "class"
        version = get_version(resource_prefix)
        cached_relabeled_obo_graph_path = prefix_directory_join(resource_prefix,
                                                            name="relabeled_obo_graph.pkl",
                                                            version=version)
        if not cached_relabeled_obo_graph_path.exists():
            download_convert_ncbitaxon_obo_to_graph()
        with open(cached_relabeled_obo_graph_path,'rb') as relabeled_graph_file:
            relabeled_graph = pickle.load(relabeled_graph_file)
    else:
        return nodes, edges

    node = relabeled_graph.nodes.get(curie)
    if not node:
        return nodes, edges
    if not add_subtree:
        property_dict = defaultdict(list)
        for text in node.get("property_value", []):
            k, v = text.split(" ", 1)
            property_dict[k].append(v)
        nodes.append(
            {
                "id": curie,
                "name": node["name"],
                "type": type,
                "description": "",
                "obsolete": False,
                "synonyms": [
                    Synonym(value=syn.split("\"")[1],
                            type="") for syn in
                    node.get("synonym", [])
                ],
                "alts": [],
                "xrefs": [Xref(id=xref_curie.lower(), type="")
                          for xref_curie in node["xref"]],
                "properties": property_dict
            }
        )
        return nodes, edges
    else:
        for node_curie in networkx.ancestors(relabeled_graph, curie) | {curie}:
            node_curie = node_curie
            node = relabeled_graph.nodes[node_curie]
            property_dict = defaultdict(list)
            for text in node.get("property_value", []):
                k, v = text.split(" ",1)
                property_dict[k].append(v)
            nodes.append(
                {
                    "id": node_curie,
                    "name": node["name"],
                    "type": type,
                    "description": "",
                    "obsolete": False,
                    "synonyms": [
                        Synonym(value=syn.split("\"")[1],
                                type="") for syn in
                        node.get("synonym", [])
                    ],
                    "alts": [],
                    "xrefs": [Xref(id=xref_curie.lower(), type="")
                              for xref_curie in node.get("xref", [])],
                    "properties": property_dict
                }
            )
            # Don't add relations where the original curie to add is the source
            # of an is_a relation. Root nodes won't have an is_a relation.
            if node_curie == curie or node["name"] == "root":
                continue
            edges.append(
                {
                    "source_curie": node_curie,
                    "target_curie": node["is_a"][0].lower(),
                    "type": is_a.name.replace(" ","_"),
                    "pred": is_a.curie,
                    "source": resource_prefix,
                    "graph": resource_prefix,
                    "version": ""
                }
            )
        return nodes, edges

@click.command()
@click.option(
    "--add-xref-edges",
    is_flag=True,
    help="Add edges for xrefs to external ontology terms",
)
@click.option(
    "--summaries",
    is_flag=True,
    help="Print summaries of nodes and edges while building",
)
@click.option("--do-upload", is_flag=True, help="Upload to S3 on completion")
@click.option("--refresh", is_flag=True, help="Refresh caches")
@click.option("--use-case", default="epi", type=click.Choice(list(cases)))
def main(
    add_xref_edges: bool,
    summaries: bool,
    do_upload: bool,
    refresh: bool,
    use_case: str,
):
    """Generate the node and edge files."""
    if Path(use_case).is_file():
        with open(use_case, 'r') as file:
            file_content = file.read()
        if use_case.lower().endswith(".json"):
            config = DKGConfig.model_validate_json(file_content)
        else:
            config = DKGConfig.model_validate(file_content)
        use_case = config.use_case
    else:
        config = None
    construct(
        use_case=use_case,
        config=config,
        refresh=refresh,
        do_upload=do_upload,
        add_xref_edges=True,
        summaries=summaries
    )


def construct(
    use_case: Optional[str] = None,
    config: Optional[DKGConfig] = None,
    *,
    refresh: bool = False,
    do_upload: bool = False,
    add_xref_edges: bool = False,
    summaries: bool = False,
):
    use_case_paths = UseCasePaths(use_case or config.use_case, config=config)

    if EDGE_NAMES_PATH.is_file():
        edge_names = json.loads(EDGE_NAMES_PATH.read_text())
    else:
        edge_names = {}
        for edge_prefix in DEFAULT_VOCABS:
            click.secho(f"Caching {manager.get_name(edge_prefix)}", fg="green", bold=True)
            parse_results = bioontologies.get_obograph_by_prefix(edge_prefix)
            for edge_graph in parse_results.graph_document.graphs:
                edge_graph = edge_graph.standardize()
                for edge_node in edge_graph.nodes:
                    if edge_node.deprecated or edge_node.id.startswith("_:genid"):
                        continue
                    if not edge_node.name:
                        if edge_node.id in LABELS:
                            edge_node.name = LABELS[edge_node.id]
                        elif edge_node.prefix:
                            edge_node.name = edge_node.identifier
                        else:
                            click.secho(f"missing label for {edge_node.curie}")
                            continue
                    if not edge_node.prefix:
                        tqdm.write(f"unparsable IRI: {edge_node.id} - {edge_node.name}")
                        continue
                    edge_names[edge_node.curie] = edge_node.name.strip()
        EDGE_NAMES_PATH.write_text(json.dumps(edge_names, sort_keys=True, indent=2))

    # A mapping from CURIEs to node information tuples
    nodes: Dict[str, NodeInfo] = {}
    # A mapping from CURIEs to a set of source strings
    node_sources = defaultdict(set)
    unstandardized_nodes = []
    unstandardized_edges = []
    edge_usage_counter = Counter()
    subject_edge_usage_counter = Counter()
    subject_edge_target_usage_counter = Counter()
    edge_target_usage_counter = Counter()

    if use_case_paths.askemo_getter is not None:
        if use_case_paths.askemo_prefix is None:
            raise ValueError
        askemo_edges = []
        click.secho(f"ASKEM custom: {use_case_paths.askemo_prefix}", fg="green", bold=True)
        for term in tqdm(use_case_paths.askemo_getter().values(), unit="term"):
            property_predicates = []
            property_values = []
            if term.suggested_unit:
                property_predicates.append("suggested_unit")
                property_values.append(term.suggested_unit)
            if term.suggested_data_type:
                property_predicates.append("suggested_data_type")
                property_values.append(term.suggested_data_type)
            if term.physical_min is not None:
                property_predicates.append("physical_min")
                property_values.append(str(term.physical_min))
            if term.physical_max is not None:
                property_predicates.append("physical_max")
                property_values.append(str(term.physical_max))
            if term.typical_min is not None:
                property_predicates.append("typical_min")
                property_values.append(str(term.typical_min))
            if term.typical_max is not None:
                property_predicates.append("typical_max")
                property_values.append(str(term.typical_max))

            node_sources[term.id].add(use_case_paths.askemo_prefix)
            nodes[term.id] = NodeInfo(
                curie=term.id,
                prefix=term.prefix,
                label=term.name,
                synonyms=";".join(synonym.value for synonym in term.synonyms or []),
                deprecated="false",
                type=term.type,
                definition=term.description,
                xrefs=";".join(xref.id for xref in term.xrefs or []),
                alts="",
                version="1.0",
                property_predicates=";".join(property_predicates),
                property_values=";".join(property_values),
                xref_types=";".join(
                    xref.type or "oboinowl:hasDbXref" for xref in term.xrefs or []
                ),
                synonym_types=";".join(
                    synonym.type or "oboInOwl:hasExactSynonym" for synonym in term.synonyms or []
                ),
            )
            for parent_curie in term.parents:
                askemo_edges.append(
                    (
                        term.id,
                        parent_curie,
                        "subclassof",
                        "rdfs:subClassOf",
                        use_case_paths.askemo_prefix,
                        use_case_paths.askemo_url,
                        "",
                    )
                )
        with use_case_paths.EDGES_PATHS[use_case_paths.askemo_prefix].open("w") as file:
            writer = csv.writer(file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(EDGE_HEADER)
            writer.writerows(askemo_edges)

    # Probability distributions
    probonto_edges = []
    for term in tqdm(get_probonto_terms(), unit="term", desc="Loading probonto"):
        curie, name, parameters = term["curie"], term["name"], term["parameters"]
        node_sources[curie].add("probonto")
        property_predicates = ["has_parameter" for _ in range(len(parameters))]
        property_values = [parameter["name"].replace("\n", " ") for parameter in parameters]
        nodes[curie] = NodeInfo(
            curie=curie,
            prefix="probonto",
            label=name,
            synonyms="",
            deprecated="false",
            type="class",
            definition="",
            xrefs=";".join(eq["curie"] for eq in term.get("equivalent", [])),
            alts="",
            version="2.5",
            property_predicates=";".join(property_predicates),
            property_values=";".join(property_values),
            xref_types=";".join("askemo:0000016" for _eq in term.get("equivalent", [])),
            synonym_types="",
        )
        # Add equivalents?
        for parameter in term.get("parameters", []):
            parameter_curie, parameter_name = parameter["curie"], parameter["name"]
            synonyms = []
            synonym_types = []
            parameter_symbol = parameter.get("symbol")
            if parameter_symbol:
                synonyms.append(parameter_symbol)
                synonym_types.append("referenced_by_latex")
            parameter_short = parameter.get("short_name")
            if parameter_short:
                synonyms.append(parameter_short)
                synonym_types.append("oboInOwl:hasExactSynonym")

            nodes[parameter_curie] = NodeInfo(
                curie=parameter_curie,
                prefix="probonto",
                label=parameter_name,
                synonyms=";".join(synonyms),
                deprecated="false",
                type="class",
                definition="",
                xrefs="",
                alts="",
                version="2.5",
                property_predicates="",
                property_values="",
                xref_types="",
                synonym_types=";".join(synonym_types),
            )
            probonto_edges.append((
                curie,
                parameter_curie,
                "has_parameter",
                "probonto:c0000062",
                "probonto",
                "https://raw.githubusercontent.com/probonto/ontology/master/probonto4ols.owl",
                "2.5",
            ))

    with use_case_paths.EDGES_PATHS["probonto"].open("w") as file:
        writer = csv.writer(file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(EDGE_HEADER)
        writer.writerows(probonto_edges)

    if use_case == "climate":

        for term in get_cso_obo().iter_terms():
            node_sources[term.curie].add("cso")
            nodes[term.curie] = get_node_info(term)

        eiffel_edges = []
        for term in tqdm(get_eiffel_ontology_terms(), unit="term", desc="Eiffel"):
            node_sources[term.curie].add("eiffel")
            nodes[term.curie] = get_node_info(term)
            for typedef, object_references in term.relationships.items():
                for object_reference in object_references:
                    eiffel_edges.append(
                        (
                            term.curie,
                            object_reference.curie,
                            typedef.name.replace(" ", "").lower(),
                            typedef.curie,
                            "eiffel",
                            "eiffel",
                            "",
                        )
                    )

        with use_case_paths.EDGES_PATHS["eiffel"].open("w") as file:
            writer = csv.writer(file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(EDGE_HEADER)
            writer.writerows(eiffel_edges)
    if use_case == "epi":
        geonames_edges = []
        for term in tqdm(get_geonames_terms(), unit="term", desc="Geonames"):
            node_sources[term.curie].add("geonames")
            nodes[term.curie] = get_node_info(term, type="individual")
            for parent in term.get_relationships(part_of):
                geonames_edges.append(
                    (
                        term.curie,
                        parent.curie,
                        "part_of",
                        part_of.curie.lower(),
                        "geonames",
                        "geonames",
                        "",
                    )
                )

        with use_case_paths.EDGES_PATHS["geonames"].open("w") as file:
            writer = csv.writer(file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(EDGE_HEADER)
            writer.writerows(geonames_edges)

        # extras from NCIT
        for term in tqdm(get_ncit_subset(), unit="term", desc="NCIT"):
            node_sources[term.curie].add("ncit")
            nodes[term.curie] = get_node_info(term, type="class")
            # TODO add edges later, if needed

        for term in tqdm(get_ncbitaxon(), unit="term", desc="NCBITaxon"):
            node_sources[term.curie].add("ncbitaxon")
            nodes[term.curie] = get_node_info(term, type="class")
            # TODO add edges to source file later, if important

    if use_case == "space":

        uat_ontology = get_uat()
        uat_edges = []
        for term in tqdm(uat_ontology, unit="term", desc="UAT"):
            node_sources[term.curie].add(uat_ontology.ontology)
            nodes[term.curie] = NodeInfo(
                curie=term.curie,
                prefix=term.prefix,
                label=term.name,
                synonyms=";".join(synonym.name for synonym in term.synonyms or []),
                deprecated="false",
                type="class",
                definition=term.definition,
                xrefs=";".join(xref.curie for xref in term.xrefs or []),
                alts="",
                version="5.0",
                property_predicates="",
                property_values="",
                xref_types="",  # TODO
                synonym_types=";".join(
                    synonym.type.curie if synonym.type is not None else "skos:exactMatch"
                    for synonym in term.synonyms or []
                ),
            )
            for parent in term.parents:
                uat_edges.append(
                    (
                        term.curie,
                        parent.curie,
                        "subclassof",
                        "rdfs:subClassOf",
                        uat_ontology.ontology,
                        uat_ontology.ontology,
                        "5.0",
                    )
                )
        with use_case_paths.EDGES_PATHS[uat_ontology.ontology].open("w") as file:
            writer = csv.writer(file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(EDGE_HEADER)
            writer.writerows(uat_edges)

    click.secho("Units", fg="green", bold=True)
    for wikidata_id, label, description, synonyms, xrefs in tqdm(get_unit_terms(), unit="unit", desc="Units"):
        curie = f"wikidata:{wikidata_id}"
        node_sources[curie].add("wikidata")
        nodes[curie] = NodeInfo(
            curie=curie,
            prefix="wikidata;unit",
            label=label,
            synonyms=";".join(synonyms),
            deprecated="false",
            type="class",
            definition=description,
            xrefs=";".join(xrefs),
            alts="",
            version="",
            property_predicates="",
            property_values="",
            xref_types=";".join("oboinowl:hasDbXref" for _ in xrefs),
            synonym_types="",
        )

    click.secho("Physical Constants", fg="green", bold=True)
    for wikidata_id, label, description, synonyms, xrefs, value, formula, symbols in tqdm(
        get_physical_constant_terms(), desc="Physical Constants"
    ):
        curie = f"wikidata:{wikidata_id}"
        node_sources[curie].add("wikidata")

        prop_predicates, prop_values = [], []
        if value:
            prop_predicates.append("debio:0000042")
            prop_values.append(str(value))
        # TODO process mathml and make readable
        # if formula:
        #     prop_predicates.append("debio:0000043")
        #     prop_values.append(str(formula))

        synonym_types, synonym_values = [], []
        for syn in synonyms:
            synonym_values.append(syn)
            synonym_types.append("oboInOwl:hasExactSynonym")
        for symbol in symbols:
            synonym_values.append(symbol)
            synonym_types.append("debio:0000031")

        nodes[curie] = NodeInfo(
            curie=curie,
            prefix="wikidata;constant",
            label=label,
            synonyms=";".join(synonym_values),
            synonym_types=";".join(synonym_types),
            deprecated="false",
            type="class",
            definition=description,
            xrefs=";".join(xrefs),
            xref_types=";".join("oboinowl:hasDbXref" for _ in xrefs),
            alts="",
            version="",
            property_predicates=";".join(prop_predicates),
            property_values=";".join(prop_values),
        )

    def _get_edge_name(curie_: str, strict: bool = False) -> str:
        if curie_ in LABELS:
            return LABELS[curie_]
        elif curie_ in edge_names:
            return edge_names[curie_]
        elif curie_ in nodes:
            return nodes[curie_][2]
        elif strict:
            raise ValueError(
                f"Can not infer name for edge curie: {curie_}. Add an entry to the LABELS dictionary"
            )
        else:
            return curie_

    biomappings_xref_graph = biomappings.get_true_graph()
    added_biomappings = 0

    for prefix in use_case_paths.prefixes:
        if prefix in {"geonames", "uat", "probonto"}:  # added with custom code
            continue
        edges = []

        _results_pickle_path = DEMO_MODULE.join("parsed", name=f"{prefix}.pkl")
        if _results_pickle_path.is_file() and not refresh:
            parse_results = pickle.loads(_results_pickle_path.read_bytes())
        else:
            if prefix in SLIMS:
                parse_results = bioontologies.get_obograph_by_path(SLIMS[prefix])
            elif _pyobo_has(prefix):
                obo = pyobo.get_ontology(prefix)
                parse_results = pyobo.parse_results_from_obo(obo)
            else:
                parse_results = bioontologies.get_obograph_by_prefix(prefix)
            if parse_results.graph_document is None:
                click.secho(
                    f"{manager.get_name(prefix)} has no graph document",
                    fg="red",
                    bold=True,
                )
                _results_pickle_path.write_bytes(pickle.dumps(parse_results))
                continue

            # Standardize graphs before caching
            parse_results.graph_document.graphs = [
                graph.standardize(tqdm_kwargs=dict(leave=False))
                for graph in tqdm(
                    parse_results.graph_document.graphs,
                    unit="graph",
                    desc=f"Standardizing graphs from {prefix}",
                    leave=False,
                )
            ]
            _results_pickle_path.write_bytes(pickle.dumps(parse_results))

        if parse_results.graph_document is None:
            click.secho(f"No graphs in {prefix}, skipping", fg="red")
            use_case_paths.EDGES_PATHS.pop(prefix)
            continue

        _graphs = parse_results.graph_document.graphs
        click.secho(
            f"{manager.get_name(prefix)} ({len(_graphs)} graphs)", fg="green", bold=True
        )
        for graph in tqdm(_graphs, unit="graph", desc=prefix, leave=False):
            graph_id = graph.id or prefix
            version = graph.version
            if version == "imports":
                version = None
            for node in graph.nodes:
                if node.deprecated or not node.reference:
                    continue
                if node.id.startswith("_:gen"):  # skip blank nodes
                    continue
                try:
                    curie = node.curie
                except ValueError:
                    tqdm.write(f"error parsing {node.id}")
                    continue
                if node.curie.startswith("_:gen"):
                    continue
                node_sources[curie].add(prefix)
                if curie not in nodes or (curie in nodes and prefix == node.prefix):
                    # TODO filter out properties that are covered elsewhere
                    properties = sorted(
                        (prop.predicate.curie, prop.value.curie)
                        for prop in node.properties
                        if prop.predicate and prop.value
                    )
                    property_predicates, property_values = [], []
                    for pred_curie, val_curie in properties:
                        property_predicates.append(pred_curie)
                        property_values.append(val_curie)

                    xref_predicates, xref_references = [], []
                    for xref in node.xrefs or []:
                        if xref.predicate and xref.value:
                            xref_predicates.append(xref.predicate.curie)
                            xref_references.append(xref.value.curie)

                    if node.curie in biomappings_xref_graph:
                        for xref_curie in biomappings_xref_graph.neighbors(node.curie):
                            if ":" not in xref_curie:
                                continue
                            added_biomappings += 1
                            xref_predicate = biomappings_xref_graph.edges[node.curie, xref_curie][
                                "relation"
                            ]
                            if xref_predicate == "speciesSpecific":
                                xref_predicate = "debio:0000003"
                            xref_predicates.append(xref_predicate)
                            xref_references.append(xref_curie)

                    nodes[curie] = NodeInfo(
                        curie=node.curie,
                        prefix=node.prefix,
                        label=node.name.strip('"')
                        .strip()
                        .strip('"')
                        .replace("\n", " ")
                        .replace("  ", " ")
                        if node.name
                        else "",
                        synonyms=";".join(synonym.value for synonym in node.synonyms),
                        deprecated="true" if node.deprecated else "false",  # type:ignore
                        # TODO better way to infer type based on hierarchy
                        #  (e.g., if rdfs:type available, consider as instance)
                        type=node.type.lower() if node.type else "unknown",  # type:ignore
                        definition=(node.definition or "")
                        .replace('"', "")
                        .replace("\n", " ")
                        .replace("  ", " "),
                        xrefs=";".join(xref_references),
                        alts=";".join(node.alternative_ids),
                        version=version or "",
                        property_predicates=";".join(property_predicates),
                        property_values=";".join(property_values),
                        xref_types=";".join(xref_predicates),
                        synonym_types=";".join(
                            synonym.predicate.curie if synonym.predicate else synonym.predicate_raw
                            for synonym in node.synonyms
                        ),
                    )

                if node.replaced_by:
                    edges.append(
                        (
                            node.replaced_by,
                            node.curie,
                            "replaced_by",
                            "iao:0100001",
                            prefix,
                            graph_id,
                            version or "",
                        )
                    )
                    if node.replaced_by not in nodes:
                        node_sources[node.replaced_by].add(prefix)
                        nodes[node.replaced_by] = NodeInfo(
                            node.replaced_by,
                            node.replaced_by.split(":", 1)[0],
                            label="",
                            synonyms="",
                            deprecated="true",
                            type="class",
                            definition="",
                            xrefs="",
                            alts="",
                            version="",
                            property_predicates="",
                            property_values="",
                            xref_types="",
                            synonym_types="",
                        )

                if add_xref_edges:
                    for xref in node.xrefs:
                        if not isinstance(xref, obograph.Xref):
                            raise TypeError(f"Invalid type: {type(xref)}: {xref}")
                        if not xref.value:
                            continue
                        if xref.value.prefix in obograph.PROVENANCE_PREFIXES:
                            # Don't add provenance information as xrefs
                            continue
                        xref_edge_info = (
                                node.curie,
                                xref.value.curie,
                                "xref",
                                "oboinowl:hasDbXref",
                                prefix,
                                graph_id,
                                version or "",
                            )
                        if xref_edge_info not in edges:
                            edges.append(xref_edge_info)
                        if xref.value.curie not in nodes:
                            node_sources[node.replaced_by].add(prefix)
                            nodes[xref.value.curie] = NodeInfo(
                                curie=xref.value.curie,
                                prefix=xref.value.prefix,
                                label="",
                                synonyms="",
                                deprecated="false",
                                type="class",
                                definition="",
                                xrefs="",
                                alts="",
                                version="",
                                property_predicates="",
                                property_values="",
                                xref_types="",
                                synonym_types="",
                            )

                for provenance in node.get_provenance():
                    if ":" in provenance.identifier:
                        tqdm.write(f"Malformed provenance for {node.curie}: {provenance}")
                    provenance_curie = provenance.curie
                    node_sources[provenance_curie].add(prefix)
                    if provenance_curie not in nodes:
                        nodes[provenance_curie] = NodeInfo(
                            curie=provenance_curie,
                            prefix=provenance.prefix,
                            label="",
                            synonyms="",
                            deprecated="false",
                            type="class",
                            definition="",
                            xrefs="",
                            alts="",
                            version="",
                            property_predicates="",
                            property_values="",
                            xref_types="",
                            synonym_types="",
                        )
                    edges.append(
                        (
                            node.curie,
                            provenance_curie,
                            "has_citation",
                            "debio:0000029",
                            prefix,
                            graph_id,
                            version or "",
                        )
                    )

            if summaries:
                counter = Counter(node.prefix for node in graph.nodes)
                tqdm.write(
                    "\n"
                    + tabulate(
                        [
                            (k, count, manager.get_name(k) if k is not None else "")
                            for k, count in counter.most_common()
                        ],
                        headers=["prefix", "count", "name"],
                        tablefmt="github",
                        # intfmt=",",
                    )
                )
                edge_counter = Counter(
                    edge.predicate.curie
                    for edge in graph.edges
                    if edge.predicate is not None
                )
                tqdm.write(
                    "\n"
                    + tabulate(
                        [
                            (pred_curie, count, _get_edge_name(pred_curie, strict=True))
                            for pred_curie, count in edge_counter.most_common()
                        ],
                        headers=["predicate", "count", "name"],
                        tablefmt="github",
                        # intfmt=",",
                    )
                    + "\n"
                )

            unstandardized_nodes.extend(node.id for node in graph.nodes if not node.reference)
            unstandardized_edges.extend(
                edge.pred for edge in graph.edges if edge.predicate is None
            )

            clean_edges = (
                edge
                for edge in graph.edges
                if (
                    edge.subject is not None
                    and edge.predicate is not None
                    and edge.object is not None
                    and edge.object.curie not in OBSOLETE
                )
            )
            edges.extend(
                (
                    edge.subject.curie,
                    edge.object.curie,
                    _get_edge_name(edge.predicate.curie).lower().replace(" ", "_").replace("-", "_"),
                    edge.predicate.curie,
                    prefix,
                    graph_id,
                    version or "",
                )
                for edge in tqdm(
                    sorted(clean_edges, key=methodcaller("as_tuple")), unit="edge", unit_scale=True
                )
            )

        for sub, obj, pred_label, pred, *_ in edges:
            edge_target_usage_counter[pred, pred_label, obj.split(":")[0]] += 1
            subject_edge_usage_counter[sub.split(":")[0], pred, pred_label] += 1
            subject_edge_target_usage_counter[
                sub.split(":")[0], pred, pred_label, obj.split(":")[0]
            ] += 1
            edge_usage_counter[pred, pred_label] += 1

        edges_path = use_case_paths.EDGES_PATHS[prefix]
        with edges_path.open("w") as file:
            writer = csv.writer(file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(EDGE_HEADER)
            writer.writerows(edges)
        tqdm.write(f"output edges to {edges_path}")

    tqdm.write(f"incorporated {added_biomappings:,} xrefs from biomappings")

    with gzip.open(use_case_paths.NODES_PATH, "wt") as file:
        writer = csv.writer(file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(NODE_HEADER)
        writer.writerows(
            (
                (*node, ";".join(sorted(node_sources[curie])))
                for curie, node in tqdm(sorted(nodes.items()), unit="node", unit_scale=True)
            )
        )
    tqdm.write(f"output edges to {use_case_paths.NODES_PATH}")

    # CAT edge files together
    with gzip.open(use_case_paths.EDGES_PATH, "wt") as file:
        writer = csv.writer(file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(EDGE_HEADER)
        for prefix, edge_path in tqdm(sorted(use_case_paths.EDGES_PATHS.items()), desc="cat edges"):
            with edge_path.open() as edge_file:
                reader = csv.reader(edge_file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                _header = next(reader)
                writer.writerows(reader)

    unstandardized_nodes_counter = Counter(unstandardized_nodes)
    _write_counter(use_case_paths.UNSTANDARDIZED_NODES_PATH, unstandardized_nodes_counter, title="url")

    unstandardized_edges_counter = Counter(unstandardized_edges)
    _write_counter(use_case_paths.UNSTANDARDIZED_EDGES_PATH, unstandardized_edges_counter, title="url")

    _write_counter(
        use_case_paths.EDGE_OBJ_COUNTER_PATH,
        edge_target_usage_counter,
        unpack=True,
        title=("predicate", "predicate_label", "object_prefix"),
    )
    _write_counter(
        use_case_paths.SUB_EDGE_COUNTER_PATH,
        subject_edge_usage_counter,
        unpack=True,
        title=("subject_prefix", "predicate", "predicate_label"),
    )
    _write_counter(
        use_case_paths.SUB_EDGE_TARGET_COUNTER_PATH,
        subject_edge_target_usage_counter,
        unpack=True,
        title=("subject_prefix", "predicate", "predicate_label", "object_prefix"),
    )
    _write_counter(
        use_case_paths.EDGE_COUNTER_PATH,
        edge_usage_counter,
        unpack=True,
        title=("predicate", "predicate_label"),
    )

    if do_upload:
        upload_neo4j_s3(use_case_paths=use_case_paths)

    from .construct_rdf import _construct_rdf

    _construct_rdf(upload=do_upload, use_case_paths=use_case_paths)

    from .construct_registry import EPI_CONF_PATH, _construct_registry

    _construct_registry(
        config_path=EPI_CONF_PATH,
        output_path=METAREGISTRY_PATH,
        upload=do_upload,
    )

    from .construct_embeddings import _construct_embeddings

    _construct_embeddings(upload=do_upload, use_case_paths=use_case_paths)

    return use_case_paths


def _write_counter(
    path: Path,
    counter: Counter,
    title: Union[None, str, Sequence[str]] = None,
    unpack: bool = False,
) -> None:
    with path.open("w") as file:
        if title:
            if unpack:
                print(*title, "count", sep="\t", file=file)
            else:
                print(title, "count", sep="\t", file=file)
        for key, count in counter.most_common():
            if unpack:
                print(*key, count, sep="\t", file=file)
            else:
                print(key, count, sep="\t", file=file)


def upload_s3(
    path: Path, *, use_case: str, bucket: str = "askem-mira", s3_client=None
) -> None:
    """Upload the nodes and edges to S3."""
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")

    today = datetime.today().strftime("%Y-%m-%d")
    # don't include a preceding or trailing slash
    key = f"dkg/{use_case}/build/{today}/"
    config = {
        # https://stackoverflow.com/questions/41904806/how-to-upload-a-file-to-s3-and-make-it-public-using-boto3
        "ACL": "public-read",
        "StorageClass": "INTELLIGENT_TIERING",
    }

    s3_client.upload_file(
        Filename=path.as_posix(),
        Bucket=bucket,
        Key=key + path.name,
        ExtraArgs=config,
    )


def upload_neo4j_s3(use_case_paths: UseCasePaths) -> None:
    """Upload the nodes and edges to S3."""
    import boto3

    s3_client = boto3.client("s3")

    paths = [
        use_case_paths.UNSTANDARDIZED_EDGES_PATH,
        use_case_paths.UNSTANDARDIZED_NODES_PATH,
        use_case_paths.NODES_PATH,
        use_case_paths.EDGES_PATH,
        use_case_paths.SUB_EDGE_COUNTER_PATH,
        use_case_paths.SUB_EDGE_TARGET_COUNTER_PATH,
        use_case_paths.EDGE_OBJ_COUNTER_PATH,
        use_case_paths.EDGE_COUNTER_PATH,
    ]
    for path in tqdm(paths):
        tqdm.write(f"uploading {path}")
        upload_s3(path=path, s3_client=s3_client, use_case=use_case_paths.use_case)


def get_node_info(term: pyobo.Term, type: EntityType = "class"):
    return NodeInfo(
        curie=term.curie,
        prefix=term.prefix,
        label=term.name,
        synonyms=";".join(synonym.name for synonym in term.synonyms or []),
        deprecated="false",
        type=type,
        definition=term.definition or "",
        xrefs="",
        alts="",
        version="",
        property_predicates="",
        property_values="",
        xref_types="",
        synonym_types=";".join(
            synonym.type.curie if synonym.type is not None else "oboInOwl:hasExactSynonym"
            for synonym in term.synonyms or []
        ),
    )


def _pyobo_has(prefix: str) -> bool:
    try:
        ontology_resolver.lookup(prefix)
    except KeyError:
        return False
    return True


if __name__ == "__main__":
    main()
