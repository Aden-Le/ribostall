import ribopy
from ribopy import Ribo
from Fasta import FastaFile
from ribopy.core.get_gadgets import get_region_boundaries, get_reference_names
import pandas as pd

def get_sequence(ribo_object, reference_file_path, alias):
    """
    Retrieves the sequences of transcripts from a reference FASTA file.

    Parameters:
        ribo_object (Ribo): The Ribo object containing ribosome profiling data.
        reference_file_path (str): The file path to the reference FASTA file.
        alias (bool): Whether or not alias is used.

    Returns:
        dict: A dictionary mapping transcript identifiers to their respective sequences.
    """
    transcript_np = ribo_object.transcript_names
    fasta = FastaFile(reference_file_path)
    
    fasta_dict = {e.header: e.sequence for e in fasta}
    if alias == True:
        sequence_dict = {
            transcript.split("|")[4]: fasta_dict[transcript] for transcript in transcript_np
        }
    else:
        sequence_dict = {
        transcript: fasta_dict[transcript] for transcript in transcript_np
    }
    return sequence_dict

def get_cds_range_lookup(ribo_object):
    """
    Create a dict of gene to CDS ranges.
    
    Parameters:
        ribo_object (Ribo): The Ribo object containing ribosome profiling data.

    Returns:
        dict: A dictionary mapping transcript identifiers to the start and stop positions of CDS.
    """
    names = get_reference_names(ribo_object._handle)
    if ribo_object.alias is not None:
        names = map(ribo_object.alias.get_alias, names)
    
    boundaries = get_region_boundaries(ribo_object._handle)
    cds_ranges = [boundary[1] for boundary in boundaries]
    boundary_lookup = dict(zip(list(names), cds_ranges))

    return boundary_lookup

def apris_human_alias(x):
    return x.split("|")[4]

from typing import Dict, Tuple, Literal
import pandas as pd

CODON_NT = 3  # one codon in nt

def get_offset(
    ribo_object,
    exp: str,
    mmin: int,
    mmax: int,
    landmark: Literal["start","stop"],
    search_window: Tuple[int,int] | None = None,
    return_site: Literal["P","A"] = "P",
) -> Dict[int,int]:
    """
    Calibrate offsets (from 5' end) per read length using metagene profiles.
    
    This function determines where the P-site (peptidyl site) or A-site (aminoacyl site) 
    is located relative to the 5' end of each read length category. It does this by:
    1. Generating metagene profiles (averaged ribosome footprint positions across all genes)
    2. Aligning genes to a specific landmark (start or stop codon)
    3. Finding the peak position (where most ribosomes are) in a search window
    4. Converting that peak position to an offset value

    Parameters:
        ribo_object: Ribo object containing the ribosome profiling data
        exp (str): Experiment name to analyze
        mmin (int): Minimum read length (nt)
        mmax (int): Maximum read length (nt)
        landmark: which reference to align metagene to ("start" or "stop").
                  This will align the genes on that specific codon, start or stop
                  Using "start" typically yields P-site offsets,
                  using "stop" typically yields A-site offsets.
        search_window: Nucleotide position range (relative to landmark) in which to search
                       for the peak. If None, uses sensible defaults.
        return_site: which site you want the final offsets for ("P" or "A").
                     We'll convert between A and P by ±3 nt as needed.
    
    Returns:
        Dict[int, int]: Dictionary mapping read length (e.g., 28) to offset value (e.g., 12)
    """
    # Set sensible default peak search windows if not provided
    # For start codon alignment: look for peak between -25 to -10 nt before start
    # For stop codon alignment: look for peak between -60 to -30 nt before stop
    if search_window is None:
        search_window = (-25, -10) if landmark == "start" else (-60, -30)

    # Get metagene profile: averaged ribosome density across all genes aligned to the landmark
    # This shows where ribosomes tend to cluster relative to start/stop codons
    # sum_lengths=False: keep data separated by read length
    # sum_references=True: combine data from all genes to get better statistics
    mg = ribo_object.get_metagene(
        site_type=landmark,
        experiments=exp,
        range_lower=mmin,
        range_upper=mmax,
        sum_lengths=False,
        sum_references=True,
    )
    # Save metagene profile to a CSV file for inspection
    mg.to_csv("metagene_profile.csv")
    print(f"[get_offset] Metagene profile saved to metagene_profile.csv")

    def _all_intlike(vals):
        """Helper: check if all values can be converted to integers."""
        try:
            pd.Series(vals).astype(int)
            return True
        except Exception:
            return False

    # Ensure columns represent integer nucleotide positions
    # The metagene dataframe might have a MultiIndex with various metadata,
    # so we need to extract the position level and use it as simple integer columns
    if isinstance(mg.columns, pd.MultiIndex):
        # Find which level contains position information
        pos_level = None
        for lvl in range(mg.columns.nlevels):
            vals = mg.columns.get_level_values(lvl)
            if _all_intlike(vals):
                pos_level = lvl
                break
        if pos_level is None:
            raise ValueError("Couldn't identify a position level in metagene columns.")
        # Extract positions and sum coverage across all non-position dimensions
        pos_vals = mg.columns.get_level_values(pos_level).astype(int)
        mg = mg.groupby(pos_vals, axis=1).sum()
    else:
        # Single index: check if it's already positions, or if we need to transpose
        if not _all_intlike(mg.columns):
            mg = mg.T
        mg.columns = mg.columns.astype(int)

    # Sort columns by position (left to right in coordinate space)
    mg = mg.loc[:, sorted(mg.columns)]

    # Group data by read length
    # The index represents read lengths (e.g., 26, 27, 28, ..., 32)
    # We'll process each length independently to calculate its specific offset
    if isinstance(mg.index, pd.MultiIndex):
        print(f"[get_offset] Metagene index is a MultiIndex with levels: {mg.index}")
        # Find which level contains read length information
        length_level = mg.index.nlevels - 1
        if not _all_intlike(mg.index.get_level_values(length_level)):
            for lvl in range(mg.index.nlevels):
                if _all_intlike(mg.index.get_level_values(lvl)):
                    length_level = lvl
                    break
        grouped = mg.groupby(level=length_level)
        print(f"[get_offset] Grouped metagene by read length using MultiIndex level {grouped}.")
    else:
        # Simple index: assume it's already read lengths
        if not _all_intlike(mg.index):
            raise ValueError("Metagene rows are not indexed by read length.")
        mg.index = mg.index.astype(int)
        grouped = ((int(L), mg.loc[[L]]) for L in mg.index.unique())
        print(f"[get_offset] Grouped metagene by read length using simple index. {grouped}")

    # Extract search window bounds
    lo, hi = search_window
    
    # Calculate offset for each read length
    offsets = {}
    # L is read length, block is the corresponding metagene data for that length
    for L, block in grouped:
        print(f"[get_offset] Processing read length L={L}")
        print(f"[get_offset]   block shape: {block.shape}, columns: {list(block.columns[:10])}...")  # Show first 10 columns
        
        # Sum coverage across all references/experiments to get 1D profile
        # This shows the aggregate ribosome density at each position
        # Basically, we want to find the position with the highest ribosome density (the peak) in our search window
        prof = block.sum(axis=0)  # 1D profile across positions
        print(f"[get_offset]   prof shape: {prof.shape}, head: {prof.head()}")  # Print first few values of the profile for sanity check
        
        # Extract only the portion of the profile within our search window
        windowed = prof.loc[(prof.index >= lo) & (prof.index <= hi)]
        
        # Skip this read length if the window is empty or has no signal
        if windowed.empty or (windowed.max() == 0):
            continue
        
        # Find the position with maximum ribosome density (the peak)
        # This peak position is relative to the landmark (start or stop codon)
        peak_pos = int(windowed.idxmax())
        print(f"[get_offset]   Peak position for L={L} is at {peak_pos} with coverage {windowed.max()}")
        
        # Convert peak position to offset from 5' end
        # Negative peak_pos means upstream of landmark, so we need to add 1 to get the 5' offset
        # Example: if peak is at -15 relative to start codon, the offset is -(-15) + 1 = 16
        base_offset = -peak_pos + 1
        offsets[int(L)] = base_offset

    # Convert offset to the requested site (P or A) if necessary
    # If landmark is "start", we typically get P-site; if "stop", we typically get A-site
    # But the user might want the opposite, so we convert using the CODON_NT (3 nt) offset
    inferred_site = "P" if landmark == "start" else "A"
    if return_site != inferred_site:
        # Convert between sites: A-site is 3 nt upstream of P-site
        # A -> P: subtract 3 (move upstream);  P -> A: add 3 (move downstream)
        delta = -CODON_NT if (inferred_site == "A" and return_site == "P") else CODON_NT
        for L in list(offsets.keys()):
            offsets[L] = int(offsets[L] + delta)

    return offsets