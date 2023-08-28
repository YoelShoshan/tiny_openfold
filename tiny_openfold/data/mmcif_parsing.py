# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parses the mmCIF file format."""
import collections
import dataclasses
import io
import json
import logging
import os
from typing import Any, Mapping, Optional, Sequence, Tuple, Dict, List

from Bio import PDB
from Bio.Data import SCOPData
import numpy as np

from tiny_openfold.data.errors import MultipleChainsError
import tiny_openfold.np.residue_constants as residue_constants
from collections import Counter

# Type aliases:
ChainId = str
PdbHeader = Mapping[str, Any]
PdbStructure = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]


@dataclasses.dataclass(frozen=True)
class Monomer:
    id: str
    num: int


# Note - mmCIF format provides no guarantees on the type of author-assigned
# sequence numbers. They need not be integers.
@dataclasses.dataclass(frozen=True)
class AtomSite:
    residue_name: str
    author_chain_id: str
    mmcif_chain_id: str
    author_seq_num: str
    mmcif_seq_num: int
    insertion_code: str
    hetatm_atom: str
    model_num: int


# Used to map SEQRES index to a residue in the structure.
@dataclasses.dataclass(frozen=True)
class ResiduePosition:
    chain_id: str
    residue_number: int
    insertion_code: str


@dataclasses.dataclass(frozen=True)
class ResidueAtPosition:
    position: Optional[ResiduePosition]
    name: str
    is_missing: bool
    hetflag: str


@dataclasses.dataclass(frozen=True)
class MmcifObject:
    """Representation of a parsed mmCIF file.

    Contains:
      file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
        files being processed.
      header: Biopython header.
      structure: Biopython structure.
      chain_to_seqres: Dict mapping *author assigned* chain_id to 1 letter amino acid sequence. E.g.
        {'A': 'ABCDEFG'}
      seqres_to_structure: Dict; for each *author assigned* chain_id contains a mapping between
        SEQRES index and a ResidueAtPosition. e.g. {'A': {0: ResidueAtPosition,
                                                          1: ResidueAtPosition,
                                                          ...}}
      raw_string: The raw string used to construct the MmcifObject.
      pdb_assigned_chain_id_to_author_assigned_chain_id: a dictionary mapping from pdb assigned chain id names to author assigned chain id
      author_assigned_chain_id_to_pdb_assigned_chain_id: a dictionary mapping from author assigned chain id names to pdb assigned chain id
    """

    file_id: str
    header: PdbHeader
    structure: PdbStructure
    chain_to_seqres: Mapping[ChainId, SeqRes]
    seqres_to_structure: Mapping[ChainId, Mapping[int, ResidueAtPosition]]
    raw_string: Any
    pdb_assigned_chain_id_to_author_assigned_chain_id: Mapping[str,str]    
    author_assigned_chain_id_to_pdb_assigned_chain_id: Mapping[str,str] 
    info: Dict  #information and statistics, useful for filtering (for example, if you want to drop cases with DNA/RNA sequences)   

@dataclasses.dataclass(frozen=True)
class ParsingResult:
    """Returned by the parse function.

    Contains:
      mmcif_object: A MmcifObject, may be None if no chain could be successfully
        parsed.
      errors: A dict mapping (file_id, chain_id) to any exception generated.
    """

    mmcif_object: Optional[MmcifObject]
    errors: Mapping[Tuple[str, str], Any]


class ParseError(Exception):
    """An error indicating that an mmCIF file could not be parsed."""


def mmcif_loop_to_list(
    prefix: str, parsed_info: MmCIFDict
) -> Sequence[Mapping[str, str]]:
    """Extracts loop associated with a prefix from mmCIF data as a list.

    Reference for loop_ in mmCIF:
      http://mmcif.wwpdb.org/docs/tutorials/mechanics/pdbx-mmcif-syntax.html

    Args:
      prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
      parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    Returns:
      Returns a list of dicts; each dict represents 1 entry from an mmCIF loop.
    """
    cols = []
    data = []
    for key, value in parsed_info.items():
        if key.startswith(prefix):
            cols.append(key)
            data.append(value)

    assert all([len(xs) == len(data[0]) for xs in data]), (
        "mmCIF error: Not all loops are the same length: %s" % cols
    )

    return [dict(zip(cols, xs)) for xs in zip(*data)]


def mmcif_loop_to_dict(
    prefix: str,
    index: str,
    parsed_info: MmCIFDict,
) -> Mapping[str, Mapping[str, str]]:
    """Extracts loop associated with a prefix from mmCIF data as a dictionary.

    Args:
      prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
      index: Which item of loop data should serve as the key.
      parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    Returns:
      Returns a dict of dicts; each dict represents 1 entry from an mmCIF loop,
      indexed by the index column.
    """
    entries = mmcif_loop_to_list(prefix, parsed_info)
    return {entry[index]: entry for entry in entries}


def parse(
    *, file_id: str, mmcif_string: str, catch_all_errors: bool = True, 
    handle_residue_id_duplication:bool = False,
    quiet_parsing:bool = True,     
    also_return_mmcif_dict:bool = False,
) -> ParsingResult:
    """Entry point, parses an mmcif_string.

    Args:
      file_id: A string identifier for this file. Should be unique within the
        collection of files being processed.
      mmcif_string: Contents of an mmCIF file.
      catch_all_errors: If True, all exceptions are caught and error messages are
        returned as part of the ParsingResult. If False exceptions will be allowed
        to propagate            
    Returns:
      A ParsingResult.
    """

    errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=quiet_parsing)
        handle = io.StringIO(mmcif_string)
        full_structure = parser.get_structure("", handle)
        first_model_structure = _get_first_model(full_structure)
        # Extract the _mmcif_dict from the parser, which contains useful fields not
        # reflected in the Biopython structure.
        parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

        # Ensure all values are lists, even if singletons.
        for key, value in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]

        header = _get_header(parsed_info)

        # Determine the protein chains, and their start numbers according to the
        # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
        valid_chains, chains_info = _get_protein_chains(parsed_info=parsed_info, 
            handle_residue_id_duplication=handle_residue_id_duplication)
        
        if not valid_chains:
            return ParsingResult(
                None, {(file_id, ""): "No protein chains found in this file."}
            )
        seq_start_num = {
            chain_id: min([monomer.num for monomer in seq])
            for chain_id, seq in valid_chains.items()
        }

        # Loop over the atoms for which we have coordinates. Populate two mappings:
        # -mmcif_to_author_chain_id (maps internal mmCIF chain ids to chain ids used
        # the authors / Biopython).
        # -seq_to_structure_mappings (maps idx into sequence to ResidueAtPosition).
        mmcif_to_author_chain_id = {}
        seq_to_structure_mappings = {}
        for atom in _get_atom_site_list(parsed_info):
            if atom.model_num != "1":
                # We only process the first model at the moment.
                continue

            if atom.mmcif_chain_id in valid_chains:

                #note: moved this into here (it was outside the check for atom.mmcif_chain_id being in valid_chains)
                mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id

                hetflag = " "
                if atom.hetatm_atom == "HETATM":
                    # Water atoms are assigned a special hetflag of W in Biopython. We
                    # need to do the same, so that this hetflag can be used to fetch
                    # a residue from the Biopython structure by id.
                    if atom.residue_name in ("HOH", "WAT"):
                        hetflag = "W"
                    else:
                        hetflag = "H_" + atom.residue_name
                insertion_code = atom.insertion_code
                if not _is_set(atom.insertion_code):
                    insertion_code = " "
                position = ResiduePosition(
                    chain_id=atom.author_chain_id,
                    residue_number=int(atom.author_seq_num),
                    insertion_code=insertion_code,
                )
                seq_idx = (
                    int(atom.mmcif_seq_num) - seq_start_num[atom.mmcif_chain_id]
                )
                current = seq_to_structure_mappings.get(
                    atom.author_chain_id, {}
                )
                current[seq_idx] = ResidueAtPosition(
                    position=position,
                    name=atom.residue_name,
                    is_missing=False,
                    hetflag=hetflag,
                )
                seq_to_structure_mappings[atom.author_chain_id] = current

        # Add missing residue information to seq_to_structure_mappings.
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            current_mapping = seq_to_structure_mappings[author_chain]
            for idx, monomer in enumerate(seq_info):
                if idx not in current_mapping:
                    current_mapping[idx] = ResidueAtPosition(
                        position=None,
                        name=monomer.id,
                        is_missing=True,
                        hetflag=" ",
                    )

        author_chain_to_sequence = {}
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            seq = []
            for monomer in seq_info:
                code = SCOPData.protein_letters_3to1.get(monomer.id, "X")
                seq.append(code if len(code) == 1 else "X")
            seq = "".join(seq)
            author_chain_to_sequence[author_chain] = seq


        author_assigned_chain_id_to_pdb_assigned_chain_id = {}
        for pdb_chain_id, author_chain_id in mmcif_to_author_chain_id.items():
            if author_chain_id in author_assigned_chain_id_to_pdb_assigned_chain_id:
                raise Exception(f'{author_chain_id} was already in author_assigned_chain_id_to_pdb_assigned_chain_id! mmcif_to_author_chain_id={mmcif_to_author_chain_id} , author_chain_ids_to_pdb_chain_ids={author_assigned_chain_id_to_pdb_assigned_chain_id}')
            author_assigned_chain_id_to_pdb_assigned_chain_id[author_chain_id] = pdb_chain_id
           
        chain_to_seqres=author_chain_to_sequence
        seqres_to_structure=seq_to_structure_mappings

        mmcif_object = MmcifObject(
            file_id=file_id,
            header=header,
            structure=first_model_structure,
            chain_to_seqres=chain_to_seqres,
            seqres_to_structure=seqres_to_structure,
            raw_string=parsed_info,
            pdb_assigned_chain_id_to_author_assigned_chain_id=mmcif_to_author_chain_id,   
            author_assigned_chain_id_to_pdb_assigned_chain_id=author_assigned_chain_id_to_pdb_assigned_chain_id,
            info = chains_info,
        )
        ans = ParsingResult(mmcif_object=mmcif_object, errors=errors)
        if not also_return_mmcif_dict:
            return ans
        else:
            return ans, parser._mmcif_dict
    except Exception as e:  # pylint:disable=broad-except
        errors[(file_id, "")] = e
        if not catch_all_errors:
            raise
        ans = ParsingResult(mmcif_object=None, errors=errors)
        if not also_return_mmcif_dict:
            return ans
        else:
            return ans, parser._mmcif_dict


def _get_first_model(structure: PdbStructure) -> PdbStructure:
    """Returns the first model in a Biopython structure."""
    return next(structure.get_models())


_MIN_LENGTH_OF_CHAIN_TO_BE_COUNTED_AS_PEPTIDE = 21


def get_release_date(parsed_info: MmCIFDict) -> str:
    """Returns the oldest revision date."""
    revision_dates = parsed_info["_pdbx_audit_revision_history.revision_date"]
    return min(revision_dates)


def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
    """Returns a basic header containing method, release date and resolution."""
    header = {}

    experiments = mmcif_loop_to_list("_exptl.", parsed_info)
    header["structure_method"] = ",".join(
        [experiment["_exptl.method"].lower() for experiment in experiments]
    )

    # Note: The release_date here corresponds to the oldest revision. We prefer to
    # use this for dataset filtering over the deposition_date.
    if "_pdbx_audit_revision_history.revision_date" in parsed_info:
        header["release_date"] = get_release_date(parsed_info)
    else:
        logging.warning(
            "Could not determine release_date: %s", parsed_info["_entry.id"]
        )

    header["resolution"] = 0.00
    for res_key in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        if res_key in parsed_info:
            try:
                raw_resolution = parsed_info[res_key][0]
                header["resolution"] = float(raw_resolution)
            except ValueError:
                logging.info(
                    "Invalid resolution format: %s", parsed_info[res_key]
                )

    return header


def _get_atom_site_list(parsed_info: MmCIFDict) -> Sequence[AtomSite]:
    """Returns list of atom sites; contains data not present in the structure."""
    return [
        AtomSite(*site)
        for site in zip(  # pylint:disable=g-complex-comprehension
            parsed_info["_atom_site.label_comp_id"],
            parsed_info["_atom_site.auth_asym_id"],
            parsed_info["_atom_site.label_asym_id"],
            parsed_info["_atom_site.auth_seq_id"],
            parsed_info["_atom_site.label_seq_id"],
            parsed_info["_atom_site.pdbx_PDB_ins_code"],
            parsed_info["_atom_site.group_PDB"],
            parsed_info["_atom_site.pdbx_PDB_model_num"],
        )
    ]

def _handle_residue_id_duplication(
    entity_id:str,
    residue_num:str,
    data:List[Dict],
    logic='keep_last'
):
    """
    Considers residue disorder (for example, point mutation) and makes sure that only a single residue is added to the aa sequence, instead of both.

    Sometimes the same residue index is mentioned twice, for example, in cases of point mutation
    it would be wrong to treat it as if the protein actually contains those 2 residues!
    an example case - pdb id 3nir
    Also, see "How is disordered handled" here https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ
    Taking the first residue for a defined index, because that seems to match the behavior in the code that extracts the coordinates.
    (TODO: should we take the first? other logic?)     
    """
    #TODO: also add 'keep_first' and other strategies
    assert logic in ['keep_last', 'keep_first']
    prev_entity_id = -1
    prev_residue_num = -1
    
    processed = []
    for entry in data:
        skip = False
        if (entry[entity_id] == prev_entity_id) and (entry[residue_num] == prev_residue_num):
            if logic == 'keep_last':
                processed = processed[:-1] #drop last (which will be replaced with current)
            elif logic == 'keep_first':
                skip = True #skip this 
            else:
                assert False, 'should not reach here'

        prev_entity_id = entry[entity_id]
        prev_residue_num = entry[residue_num]

        if skip:
            continue

        processed.append(entry)
    return processed

def _get_protein_chains(
    *, parsed_info: Mapping[str, Any],
    handle_residue_id_duplication:bool = False,
) -> Mapping[ChainId, Sequence[Monomer]]:
    """Extracts polymer information for protein chains only.

    Args:
      parsed_info: _mmcif_dict produced by the Biopython parser.
      handle_residue_id_duplication: handles residue id duplication.
        Which happens, for example, in point mutation. see pdb id 3nir as an example case
        See handle_residue_id_duplication doc string for more details

    Returns:
      A dict mapping mmcif chain id to a list of Monomers.
    """
    # Get polymer information for each entity in the structure.
    entity_poly_seqs = mmcif_loop_to_list("_entity_poly_seq.", parsed_info)

    if handle_residue_id_duplication:
        entity_poly_seqs = _handle_residue_id_duplication(
            entity_id = '_entity_poly_seq.entity_id',
            residue_num = '_entity_poly_seq.num',
            data = entity_poly_seqs,
            logic = 'keep_last',
        )

    polymers = collections.defaultdict(list)
    for entity_poly_seq in entity_poly_seqs:
        polymers[entity_poly_seq["_entity_poly_seq.entity_id"]].append(
            Monomer(
                id=entity_poly_seq["_entity_poly_seq.mon_id"],
                num=int(entity_poly_seq["_entity_poly_seq.num"]),
            )
        )

    # Get chemical compositions. Will allow us to identify which of these polymers
    # are proteins.
    chem_comps = mmcif_loop_to_dict("_chem_comp.", "_chem_comp.id", parsed_info)

    # Get chains information for each entity. Necessary so that we can return a
    # dict keyed on chain id rather than entity.
    struct_asyms = mmcif_loop_to_list("_struct_asym.", parsed_info)

    entity_to_mmcif_chains = collections.defaultdict(list)
    for struct_asym in struct_asyms:
        chain_id = struct_asym["_struct_asym.id"]
        entity_id = struct_asym["_struct_asym.entity_id"]
        entity_to_mmcif_chains[entity_id].append(chain_id)

    # Identify and return the valid protein chains.
    valid_chains = {}
    info = {}

    _chem_comp_type_total_stats_counter = Counter()
    
    rna_or_dna_only_sequences_count = 0
    for entity_id, seq_info in polymers.items():
        chain_ids = entity_to_mmcif_chains[entity_id]

        # Reject polymers without any peptide-like components, such as DNA/RNA.

        components = [ chem_comps[monomer.id]["_chem_comp.type"].lower() for monomer in seq_info ]

        for comp in components:
            _chem_comp_type_total_stats_counter[comp] += 1

        peptide_found = any(['peptide' in x for x in components])
        rna_or_dna_only_sequences_count += any([ 
           (('rna' in x) or ('dna' in x))
           for x in components])

        if peptide_found:
            for chain_id in chain_ids:
                valid_chains[chain_id] = seq_info

    info['_chem_comp_type_total_stats_counter'] = _chem_comp_type_total_stats_counter
    info['rna_or_dna_only_sequences_count'] = rna_or_dna_only_sequences_count
        
    return valid_chains, info


def _is_set(data: str) -> bool:
    """Returns False if data is a special mmCIF character indicating 'unset'."""
    return data not in (".", "?")


def get_atom_coords(
    mmcif_object: MmcifObject, 
    chain_id: str, 
    _zero_center_positions: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    # Locate the right chain
    chains = list(mmcif_object.structure.get_chains())
    relevant_chains = [c for c in chains if c.id == chain_id]
    if len(relevant_chains) != 1:
        raise MultipleChainsError(
            f"Expected exactly one chain in structure with id {chain_id}, but instead got {len(relevant_chains)}."
        )
    chain = relevant_chains[0]

    # Extract the coordinates
    num_res = len(mmcif_object.chain_to_seqres[chain_id])
    all_atom_positions = np.zeros(
        [num_res, residue_constants.atom_type_num, 3], dtype=np.float32
    )
    all_atom_mask = np.zeros(
        [num_res, residue_constants.atom_type_num], dtype=np.float32
    )
    for res_index in range(num_res):
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        res_at_position = mmcif_object.seqres_to_structure[chain_id][res_index]
        if not res_at_position.is_missing:
            res = chain[
                (
                    res_at_position.hetflag,
                    res_at_position.position.residue_number,
                    res_at_position.position.insertion_code,
                )
            ]
            for atom in res.get_atoms():
                atom_name = atom.get_name()
                x, y, z = atom.get_coord()
                if atom_name in residue_constants.atom_order.keys():
                    pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                    mask[residue_constants.atom_order[atom_name]] = 1.0
                elif atom_name.upper() == "SE" and res.get_resname() == "MSE":
                    # Put the coords of the selenium atom in the sulphur column
                    pos[residue_constants.atom_order["SD"]] = [x, y, z]
                    mask[residue_constants.atom_order["SD"]] = 1.0

        all_atom_positions[res_index] = pos
        all_atom_mask[res_index] = mask

    if _zero_center_positions:
        binary_mask = all_atom_mask.astype(bool)
        translation_vec = all_atom_positions[binary_mask].mean(axis=0)
        all_atom_positions[binary_mask] -= translation_vec

    return all_atom_positions, all_atom_mask
