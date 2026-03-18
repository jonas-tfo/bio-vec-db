
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum SeqType {
    Dna,
    Rna,
    Protein,
}

/// record for the sled storage
#[derive(Serialize, Deserialize, Debug)]
pub struct FastaRecord {
    pub header: String,
    pub sequence: Vec<u8>,  // raw seq as vec of bytes
    pub seq_type: SeqType,
}

pub struct Storage {
    pub db: sled::Db,
    pub records: sled::Tree,  // tree for FastaRecord entries
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SequenceEmbedder {
    Esm2_650M,      // Protein
    DnaBert2,       // DNA
    ProtBert,       // Protein
    ProtT5,         // Protein
    Custom(String), // user defined
}

