"""Curated gene signatures for the gene-list queries (E9).

TISSUE_MODULES: per-tissue signatures; every gene was checked against that tissue's Xenium panel.
GENERIC_MODULES: cross-tissue pathway signatures (genes missing from a panel are dropped).
benchmarks/biology/signature_queries.py uses TISSUE_MODULES (one entry per tissue).
"""

# Tissue-specific curated gene signatures. Every gene listed here was verified to be
# present in the corresponding Xenium panel before being added.
TISSUE_MODULES = {
    "breast": {
        "Luminal_Tumor_Core": {
            "description": "Invasive Tumor & DCIS Epithelial Core",
            "genes": ["ESR1", "PGR", "ERBB2", "FOXA1", "GATA3", "KRT8", "EPCAM", "CDH1"],
            "target_clusters": ["Invasive_Tumor", "DCIS_1", "DCIS_2"],
            "color": "#E64B35"
        },
        "Macrophage_Myeloid": {
            "description": "Macrophage & Dendritic Cell Niche",
            "genes": ["CD68", "CD163", "MRC1", "C1QA", "ITGAX"],
            "target_clusters": ["Macrophages_1", "Macrophages_2", "IRF7+_DCs", "LAMP3+_DCs"],
            "color": "#00A087"
        },
        "Basal_Myoepithelial": {
            "description": "Myoepithelial & Basal Layer",
            "genes": ["KRT5", "KRT14", "ACTA2", "MYLK"],
            "target_clusters": ["Myoepi_ACTA2+", "Myoepi_KRT15+"],
            "color": "#F39B7F"
        },
        "Endothelial_Vascular": {
            "description": "Vasculature & Endothelial Cells",
            "genes": ["PECAM1", "VWF", "KDR"],
            "target_clusters": ["Endothelial", "Perivascular-Like"],
            "color": "#3C5488"
        },
        "Proliferation_Signature": {
            "description": "Actively Proliferating Tumor Cells",
            "genes": ["MKI67", "TOP2A"],
            "target_clusters": ["Prolif_Invasive_Tumor"],
            "color": "#8491B4"
        }
    },
    "kidney": {
        "Tubular_Epithelium": {
            "description": "Proximal/Distal Tubule & Collecting Duct Segments",
            "genes": ["SLC22A8", "SLC4A1", "AQP2", "UMOD", "SLC26A3"],
            "target_clusters": [],
            "color": "#4DBBD5"
        },
        "Injury_Immune_Infiltrate": {
            "description": "Tubular Injury Marker & Immune Infiltrate",
            "genes": ["HAVCR2", "CD68", "CD163", "CD3D", "CD3E"],
            "target_clusters": [],
            "color": "#91D1C2"
        },
    },
    "lung": {
        "Alveolar_Airway_Epithelium": {
            "description": "Alveolar Type 1 & Ciliated/Secretory Airway Epithelium",
            "genes": ["AGER", "FOXJ1", "SCGB2A1", "KRT7", "SOX2"],
            "target_clusters": [],
            "color": "#F39B7F"
        },
        "Vascular_Endothelial": {
            "description": "Lung Vasculature & Endothelium",
            "genes": ["SOX17", "SOX18", "PECAM1", "VWF"],
            "target_clusters": [],
            "color": "#3C5488"
        },
    },
    "pancrea": {
        "Exocrine_Ductal": {
            "description": "Ductal & Exocrine Epithelium",
            "genes": ["KRT7", "KRT20", "CFTR"],
            "target_clusters": [],
            "color": "#E64B35"
        },
        "Endocrine_Islet": {
            "description": "Islets of Langerhans (Alpha/Beta/Delta/PP Cells)",
            "genes": ["INS", "GCG", "SST", "PPY", "CHGA"],
            "target_clusters": [],
            "color": "#00A087"
        },
        "Mast_Cell_Stroma": {
            "description": "Mast Cells & Perivascular Stroma",
            "genes": ["CPA3", "ACTA2", "PDGFRB"],
            "target_clusters": [],
            "color": "#8491B4"
        },
    },
    "brain": {
        "Glioma_Tumor_Proliferative": {
            "description": "Proliferating Glioma Tumor Core",
            "genes": ["EGFR", "PDGFRA", "MKI67"],
            "target_clusters": [],
            "color": "#E64B35"
        },
        "Microglia_Myeloid": {
            "description": "Microglia & Tumor-Associated Myeloid Cells",
            "genes": ["CD68", "CD44", "PTPRC", "MMP9", "MMP12"],
            "target_clusters": [],
            "color": "#00A087"
        },
        "TIL_Lymphocyte": {
            "description": "Tumor-Infiltrating Lymphocytes",
            "genes": ["CD3D", "CD3E", "CD37", "CD38"],
            "target_clusters": [],
            "color": "#4DBBD5"
        },
    },
    "lymph_node": {
        "B_Cell_Germinal_Center": {
            "description": "B Cell / Germinal Center Niche",
            "genes": ["MS4A1", "CD79A", "CD19", "MZB1"],
            "target_clusters": [],
            "color": "#00A087"
        },
        "T_Cell_Zone": {
            "description": "Paracortical T Cell Zone",
            "genes": ["CD3D", "CD3E", "CD4", "CD8A"],
            "target_clusters": [],
            "color": "#4DBBD5"
        },
        "Macrophage_DC_Niche": {
            "description": "Macrophage & Dendritic Cell Niche",
            "genes": ["CD68", "CD163", "LAMP3", "CD83", "CD86", "CD300E"],
            "target_clusters": [],
            "color": "#F39B7F"
        },
        "Lymphatic_Vascular_Stroma": {
            "description": "Lymphatic Sinuses & Vascular Stroma",
            "genes": ["LYVE1", "PROX1", "PDPN", "CD34", "CCL19"],
            "target_clusters": [],
            "color": "#3C5488"
        },
    },
    "skin": {
        "Melanocyte_Tumor": {
            "description": "Melanocyte / Melanoma Tumor Core",
            "genes": ["TYR", "TYRP1", "MLANA", "PMEL", "DCT", "MITF", "SOX10", "S100B"],
            "target_clusters": [],
            "color": "#E64B35"
        },
        "Keratinocyte_Epidermis": {
            "description": "Epidermal Keratinocyte Layers",
            "genes": ["KRT1", "KRT5", "KRT14", "KRT17", "LOR"],
            "target_clusters": [],
            "color": "#F39B7F"
        },
        "Immune_Infiltrate": {
            "description": "Immune Cell Infiltrate",
            "genes": ["CD3D", "CD3E", "CD68", "C1QA", "ITGAX"],
            "target_clusters": [],
            "color": "#00A087"
        },
    },
}

# Cross-tissue functional/pathway modules, run identically on every dataset. Genes
# missing from a given panel are dropped automatically by find_best_matching_block;
# the resulting per-dataset overlap differences are themselves part of the signal.
GENERIC_MODULES = {
    "Cytotoxic_Immune": {
        "description": "Cytotoxic T / NK Cell Activity",
        "genes": ["NKG7", "GNLY", "KLRD1", "CD8A"],
        "target_clusters": [],
        "color": "#8491B4"
    },
    "Pan_Myeloid_Macrophage": {
        "description": "Pan-Myeloid / Macrophage Compartment",
        "genes": ["CD68", "CD163", "CD14", "FCGR3A", "ITGAX"],
        "target_clusters": [],
        "color": "#91D1C2"
    },
    "Interferon_Response": {
        "description": "Interferon-Stimulated Response",
        "genes": ["CXCL9", "CXCL10", "CXCL11", "STAT1", "IRF1"],
        "target_clusters": [],
        "color": "#7E6148"
    },
    "Inflammatory_Response": {
        "description": "General Inflammatory / Cytokine Response",
        "genes": ["IL6", "TNF", "IL1B", "CCL2", "ICAM1"],
        "target_clusters": [],
        "color": "#B09C85"
    },
    "Hypoxia_Angiogenesis": {
        "description": "Hypoxia-Driven Angiogenic Signaling",
        "genes": ["VEGFA", "HIF1A", "MMP9"],
        "target_clusters": [],
        "color": "#DC0000"
    },
    "EMT_Stromal_Invasion": {
        "description": "Epithelial-Mesenchymal Transition & Stromal Invasion",
        "genes": ["VIM", "CDH2", "SNAI1", "ZEB1", "FN1", "MMP2", "SPARC"],
        "target_clusters": [],
        "color": "#631879"
    },
}
