import os
import json

category_mapper = {'GTEX Adipose Tissue':                       'GTEX-ADI',
                   'GTEX Adrenal Gland':                        'GTEX-ADR_GLA',
                   'GTEX Bladder':                              'GTEX-BLA',
                   'GTEX Blood':                                'GTEX-BLO',
                   'GTEX Blood Vessel':                         'GTEX-BLO_VSL',
                   'GTEX Brain':                                'GTEX-BRA',
                   'GTEX Breast':                               'GTEX-BRE',
                   'GTEX Cervix Uteri':                         'GTEX-CER',
                   'GTEX Colon':                                'GTEX-COL',
                   'GTEX Esophagus':                            'GTEX-ESO',
                   'GTEX Fallopian Tube':                       'GTEX-FAL_TUB',
                   'GTEX Heart':                                'GTEX-HEA',
                   'GTEX Kidney':                               'GTEX-KID',
                   'GTEX Liver':                                'GTEX-LIV',
                   'GTEX Lung':                                 'GTEX-LUN',
                   'GTEX Muscle':                               'GTEX-MUS',
                   'GTEX Nerve':                                'GTEX-NER',
                   'GTEX Ovary':                                'GTEX-OVA',
                   'GTEX Pancreas':                             'GTEX-PAN',
                   'GTEX Pituitary':                            'GTEX-PIT',
                   'GTEX Prostate':                             'GTEX-PRO',
                   'GTEX Salivary Gland':                       'GTEX-SAL_GLA',
                   'GTEX Skin':                                 'GTEX-SKI',
                   'GTEX Small Intestine':                      'GTEX-SMA_INT',
                   'GTEX Spleen':                               'GTEX-SPL',
                   'GTEX Stomach':                              'GTEX-STO',
                   'GTEX Testis':                               'GTEX-TES',
                   'GTEX Thyroid':                              'GTEX-THY',
                   'GTEX Uterus':                               'GTEX-UTE',
                   'GTEX Vagina':                               'GTEX-VAG',
                   'TCGA Acute Myeloid Leukemia':               'TCGA-LAML',
                   'TCGA Adrenocortical Cancer':                'TCGA-ACC',
                   'TCGA Bladder Urothelial Carcinoma':         'TCGA-BLCA',
                   'TCGA Brain Lower Grade Glioma':             'TCGA-LGG',
                   'TCGA Breast Invasive Carcinoma':            'TCGA-BRCA',
                   'TCGA Cervical & Endocervical Cancer':       'TCGA-CESC',
                   'TCGA Cholangiocarcinoma':                   'TCGA-CHOL',
                   'TCGA Colon Adenocarcinoma':                 'TCGA-COAD',
                   'TCGA Diffuse Large B-Cell Lymphoma':        'TCGA-DLBC',
                   'TCGA Esophageal Carcinoma':                 'TCGA-ESCA',
                   'TCGA Glioblastoma Multiforme':              'TCGA-GBM',
                   'TCGA Head & Neck Squamous Cell Carcinoma':  'TCGA-HNSC',
                   'TCGA Kidney Chromophobe':                   'TCGA-KICH',
                   'TCGA Kidney Clear Cell Carcinoma':          'TCGA-KIRC',
                   'TCGA Kidney Papillary Cell Carcinoma':      'TCGA-KIRP',
                   'TCGA Liver Hepatocellular Carcinoma':       'TCGA-LIHC',
                   'TCGA Lung Adenocarcinoma':                  'TCGA-LUAD',
                   'TCGA Lung Squamous Cell Carcinoma':         'TCGA-LUSC',
                   'TCGA Mesothelioma':                         'TCGA-MESO',
                   'TCGA Ovarian Serous Cystadenocarcinoma':    'TCGA-OV',
                   'TCGA Pancreatic Adenocarcinoma':            'TCGA-PAAD',
                   'TCGA Pheochromocytoma & Paraganglioma':     'TCGA-PCPG',
                   'TCGA Prostate Adenocarcinoma':              'TCGA-PRAD',
                   'TCGA Rectum Adenocarcinoma':                'TCGA-READ',
                   'TCGA Sarcoma':                              'TCGA-SARC',
                   'TCGA Skin Cutaneous Melanoma':              'TCGA-SKCM',
                   'TCGA Stomach Adenocarcinoma':               'TCGA-STAD',
                   'TCGA Testicular Germ Cell Tumor':           'TCGA-TGCT',
                   'TCGA Thymoma':                              'TCGA-THYM',
                   'TCGA Thyroid Carcinoma':                    'TCGA-THCA',
                   'TCGA Uterine Carcinosarcoma':               'TCGA-UCS',
                   'TCGA Uterine Corpus Endometrioid Carcinoma':'TCGA-UCEC',
                   'TCGA Uveal Melanoma':                       'TCGA-UVM'
                   }


phenotype_mapper = {'Adipose - Subcutaneous':                       'GTEX-ADI', 
                    'Adipose - Visceral (Omentum)':                 'GTEX-ADI',
                    'Adrenal Gland':                                'GTEX-ADR_GLA',
                    'Artery - Aorta':                               'GTEX-BLO_VSL',
                    'Artery - Coronary':                            'GTEX-BLO_VSL',
                    'Artery - Tibial':                              'GTEX-BLO_VSL',
                    'Bladder':                                      'GTEX-BLA',
                    'Brain - Amygdala':                             'GTEX-BRA',
                    'Brain - Anterior Cingulate Cortex (Ba24)':     'GTEX-BRA',
                    'Brain - Caudate (Basal Ganglia)':              'GTEX-BRA',
                    'Brain - Cerebellar Hemisphere':                'GTEX-BRA',
                    'Brain - Cerebellum':                           'GTEX-BRA',
                    'Brain - Cortex':                               'GTEX-BRA',
                    'Brain - Frontal Cortex (Ba9)':                 'GTEX-BRA',
                    'Brain - Hippocampus':                          'GTEX-BRA',
                    'Brain - Hypothalamus':                         'GTEX-BRA',
                    'Brain - Nucleus Accumbens (Basal Ganglia)':    'GTEX-BRA',
                    'Brain - Putamen (Basal Ganglia)':              'GTEX-BRA',
                    'Brain - Spinal Cord (Cervical C-1)':           'GTEX-BRA',
                    'Brain - Substantia Nigra':                     'GTEX-BRA',
                    'Breast - Mammary Tissue':                      'GTEX-BRE',
                    'Cells - Ebv-Transformed Lymphocytes':          'GTEX-BLO',
                    'Cells - Leukemia Cell Line (Cml)':             'TCGA-LAML', # TODO: Know what to do with this category. This is a problematic category because it is GTEX but appears to be of sick patients
                    'Cells - Transformed Fibroblasts':              'GTEX-SKI',
                    'Cervix - Ectocervix':                          'GTEX-CER',
                    'Cervix - Endocervix':                          'GTEX-CER',
                    'Colon - Sigmoid':                              'GTEX-COL',
                    'Colon - Transverse':                           'GTEX-COL',
                    'Esophagus - Gastroesophageal Junction':        'GTEX-ESO',
                    'Esophagus - Mucosa':                           'GTEX-ESO',
                    'Esophagus - Muscularis':                       'GTEX-ESO',
                    'Fallopian Tube':                               'GTEX-FAL_TUB',
                    'Heart - Atrial Appendage':                     'GTEX-HEA',
                    'Heart - Left Ventricle':                       'GTEX-HEA',
                    'Kidney - Cortex':                              'GTEX-KID',
                    'Liver':                                        'GTEX-LIV',
                    'Lung':                                         'GTEX-LUN',
                    'Minor Salivary Gland':                         'GTEX-SAL_GLA',
                    'Muscle - Skeletal':                            'GTEX-MUS',
                    'Nerve - Tibial':                               'GTEX-NER',
                    'Ovary':                                        'GTEX-OVA',
                    'Pancreas':                                     'GTEX-PAN',
                    'Pituitary':                                    'GTEX-PIT',
                    'Prostate':                                     'GTEX-PRO',
                    'Skin - Not Sun Exposed (Suprapubic)':          'GTEX-SKI',
                    'Skin - Sun Exposed (Lower Leg)':               'GTEX-SKI',
                    'Small Intestine - Terminal Ileum':             'GTEX-SMA_INT',
                    'Spleen':                                       'GTEX-SPL',
                    'Stomach':                                      'GTEX-STO',
                    'Testis':                                       'GTEX-TES',
                    'Thyroid':                                      'GTEX-THY',
                    'Uterus':                                       'GTEX-UTE',
                    'Vagina':                                       'GTEX-VAG',
                    'Whole Blood':                                  'GTEX-BLO',
                    'Acute Myeloid Leukemia':                       'TCGA-LAML',
                    'Adrenocortical Cancer':                        'TCGA-ACC',
                    'Bladder Urothelial Carcinoma':                 'TCGA-BLCA',
                    'Brain Lower Grade Glioma':                     'TCGA-LGG',
                    'Breast Invasive Carcinoma':                    'TCGA-BRCA',
                    'Cervical & Endocervical Cancer':               'TCGA-CESC',
                    'Cholangiocarcinoma':                           'TCGA-CHOL',
                    'Colon Adenocarcinoma':                         'TCGA-COAD',
                    'Diffuse Large B-Cell Lymphoma':                'TCGA-DLBC',
                    'Esophageal Carcinoma':                         'TCGA-ESCA',
                    'Glioblastoma Multiforme':                      'TCGA-GBM',
                    'Head & Neck Squamous Cell Carcinoma':          'TCGA-HNSC',
                    'Kidney Chromophobe':                           'TCGA-KICH',
                    'Kidney Clear Cell Carcinoma':                  'TCGA-KIRC',
                    'Kidney Papillary Cell Carcinoma':              'TCGA-KIRP',
                    'Liver Hepatocellular Carcinoma':               'TCGA-LIHC',
                    'Lung Adenocarcinoma':                          'TCGA-LUAD',
                    'Lung Squamous Cell Carcinoma':                 'TCGA-LUSC',
                    'Mesothelioma':                                 'TCGA-MESO',
                    'Ovarian Serous Cystadenocarcinoma':            'TCGA-OV',
                    'Pancreatic Adenocarcinoma':                    'TCGA-PAAD',
                    'Pheochromocytoma & Paraganglioma':             'TCGA-PCPG',
                    'Prostate Adenocarcinoma':                      'TCGA-PRAD',
                    'Rectum Adenocarcinoma':                        'TCGA-READ',
                    'Sarcoma':                                      'TCGA-SARC',
                    'Skin Cutaneous Melanoma':                      'TCGA-SKCM',
                    'Stomach Adenocarcinoma':                       'TCGA-STAD',
                    'Testicular Germ Cell Tumor':                   'TCGA-TGCT',
                    'Thymoma':                                      'TCGA-THYM',
                    'Thyroid Carcinoma':                            'TCGA-THCA',
                    'Uterine Carcinosarcoma':                       'TCGA-UCS',
                    'Uterine Corpus Endometrioid Carcinoma':        'TCGA-UCEC',
                    'Uveal Melanoma':                               'TCGA-UVM'
                    }
# There are a total of 93/738 problematic samples
normal_tcga_mapper = {  'Bile duct':            'GTEX-LIV',     # Problematic 9 samples
                        'Bladder':              'GTEX-BLA',
                        'Breast':               'GTEX-BRE',
                        'Cervix':               'GTEX-CER',
                        'Colon':                'GTEX-COL',
                        'Endometrium':          'GTEX-UTE',     # Problematic 23 samples
                        'Esophagus':            'GTEX-ESO',
                        'Head and Neck region': 'GTEX-SKI',     # Problematic 44 samples (Squamous cells are found in the outer layer of skin and in the mucous membranes)
                        'Kidney':               'GTEX-KID',
                        'Liver':                'GTEX-LIV',
                        'Lung':                 'GTEX-LUN',
                        'Pancreas':             'GTEX-PAN',
                        'Paraganglia':          'GTEX-ADR_GLA', # Problematic 3 samples
                        'Prostate':             'GTEX-PRO',
                        'Rectum':               'GTEX-COL',     # Problematic 10 samples
                        'Skin':                 'GTEX-SKI',
                        'Soft tissue,Bone':     'GTEX-ADI',     # Problematic 2 samples
                        'Stomach':              'GTEX-STO',
                        'Thymus':               'GTEX-SKI',     # Problematic 2 samples (Thymoma and thymic carcinoma, also called thymic epithelial tumors (TETs))
                        'Thyroid Gland':        'GTEX-THY'
                        }

# Define a mapper from identifiers to tissue specific models
id_2_tissue_mapper = {  'GTEX-ADI':             'Connective', # Model for connective tissue to compare with sarcoma
                        'GTEX-ADR_GLA':         'Kidney', 
                        'GTEX-BLA':             'Bladder',
                        'GTEX-BLO':             'Blood',
                        'GTEX-BLO_VSL':         'Blood',
                        'GTEX-BRA':             'Brain',
                        'GTEX-BRE':             'Breast',
                        'GTEX-CER':             'Cervix',
                        'GTEX-COL':             'Colon',
                        'GTEX-ESO':             'Esophagus',
                        'GTEX-FAL_TUB':         'Not Paired', # Fallopian tube
                        'GTEX-HEA':             'Not Paired', # Heart
                        'GTEX-KID':             'Kidney',
                        'GTEX-LIV':             'Liver',
                        'GTEX-LUN':             'Lung',
                        'GTEX-MUS':             'Connective',
                        'GTEX-NER':             'Not Paired', # Nerve
                        'GTEX-OVA':             'Ovary',
                        'GTEX-PAN':             'Pancreas',
                        'GTEX-PIT':             'Brain',
                        'GTEX-PRO':             'Prostate',
                        'GTEX-SAL_GLA':         'Not Paired', # Salivary gland
                        'GTEX-SKI':             'Skin',
                        'GTEX-SMA_INT':         'Not Paired', # Small intestine
                        'GTEX-SPL':             'Not Paired', # Spleen
                        'GTEX-STO':             'Stomach',
                        'GTEX-TES':             'Testis',
                        'GTEX-THY':             'Thyroid',
                        'GTEX-UTE':             'Uterus',
                        'GTEX-VAG':             'Not Paired', # Vagina
                        'TCGA-LAML':            'Blood',
                        'TCGA-ACC':             'Kidney',
                        'TCGA-BLCA':            'Bladder',
                        'TCGA-LGG':             'Brain',
                        'TCGA-BRCA':            'Breast',
                        'TCGA-CESC':            'Cervix',
                        'TCGA-CHOL':            'Liver',
                        'TCGA-COAD':            'Colon',
                        'TCGA-DLBC':            'Not Paired', # Difuse large B-cell lymphoma is from the linphatic system but assigned to blood
                        'TCGA-ESCA':            'Esophagus',
                        'TCGA-GBM':             'Brain',
                        'TCGA-HNSC':            'Not Paired', # Head and neck squamous cell carcinoma
                        'TCGA-KICH':            'Kidney',
                        'TCGA-KIRC':            'Kidney',
                        'TCGA-KIRP':            'Kidney',
                        'TCGA-LIHC':            'Liver',
                        'TCGA-LUAD':            'Lung',
                        'TCGA-LUSC':            'Lung',
                        'TCGA-MESO':            'Lung',
                        'TCGA-OV':              'Ovary',
                        'TCGA-PAAD':            'Pancreas',
                        'TCGA-PCPG':            'Kidney',
                        'TCGA-PRAD':            'Prostate',
                        'TCGA-READ':            'Colon',
                        'TCGA-SARC':            'Connective',
                        'TCGA-SKCM':            'Skin',
                        'TCGA-STAD':            'Stomach',
                        'TCGA-TGCT':            'Testis',
                        'TCGA-THYM':            'Not Paired', # Thymoma
                        'TCGA-THCA':            'Thyroid',
                        'TCGA-UCS':             'Uterus',
                        'TCGA-UCEC':            'Uterus',
                        'TCGA-UVM':             'Not Paired'} # Uveal melanoma


wang_standard_label_mapper = {  'GTEX-BLADDER':         'GTEX-BLA',
                                'GTEX-BREAST':          'GTEX-BRE',
                                'GTEX-CERVIX':          'GTEX-CER',
                                'GTEX-COLON':           'GTEX-COL',
                                'GTEX-ESOPHAGUS_GAS':   'GTEX-ESO',
                                'GTEX-ESOPHAGUS_MUC':   'GTEX-ESO',
                                'GTEX-ESOPHAGUS_MUS':   'GTEX-ESO',
                                'GTEX-KIDNEY':          'GTEX-KID',
                                'GTEX-LIVER':           'GTEX-LIV',
                                'GTEX-LUNG':            'GTEX-LUN',
                                'GTEX-PROSTATE':        'GTEX-PRO',
                                'GTEX-SALIVARY':        'GTEX-SAL_GLA',
                                'GTEX-STOMACH':         'GTEX-STO',
                                'GTEX-THYROID':         'GTEX-THY',
                                'GTEX-UTERUS':          'GTEX-UTE',
                                'TCGA-BLCA':            'GTEX-BLA',
                                'TCGA-BRCA':            'GTEX-BRE',
                                'TCGA-CESC':            'GTEX-CER',
                                'TCGA-COAD':            'GTEX-COL',
                                'TCGA-ESCA':            'GTEX-ESO',
                                'TCGA-HNSC':            'GTEX-SAL_GLA',
                                'TCGA-KICH':            'GTEX-KID',
                                'TCGA-KIRC':            'GTEX-KID',
                                'TCGA-KIRP':            'GTEX-KID',
                                'TCGA-LIHC':            'GTEX-LIV',
                                'TCGA-LUAD':            'GTEX-LUN',
                                'TCGA-LUSC':            'GTEX-LUN',
                                'TCGA-PRAD':            'GTEX-PRO',
                                'TCGA-READ':            'GTEX-COL',
                                'TCGA-STAD':            'GTEX-STO',
                                'TCGA-THCA':            'GTEX-THY',
                                'TCGA-UCEC':            'GTEX-UTE',
                                'TCGA-T-BLCA':          'TCGA-BLCA',
                                'TCGA-T-BRCA':          'TCGA-BRCA',
                                'TCGA-T-CESC':          'TCGA-CESC',
                                'TCGA-T-COAD':          'TCGA-COAD',
                                'TCGA-T-ESCA':          'TCGA-ESCA',
                                'TCGA-T-HNSC':          'TCGA-HNSC',
                                'TCGA-T-KICH':          'TCGA-KICH',
                                'TCGA-T-KIRC':          'TCGA-KIRC',
                                'TCGA-T-KIRP':          'TCGA-KIRP',
                                'TCGA-T-LIHC':          'TCGA-LIHC',
                                'TCGA-T-LUAD':          'TCGA-LUAD',
                                'TCGA-T-LUSC':          'TCGA-LUSC',
                                'TCGA-T-PRAD':          'TCGA-PRAD',
                                'TCGA-T-READ':          'TCGA-READ',
                                'TCGA-T-STAD':          'TCGA-STAD',
                                'TCGA-T-THCA':          'TCGA-THCA',
                                'TCGA-T-UCEC':          'TCGA-UCEC',
                                'TCGA-T-UCS':           'TCGA-UCS'}


recount3_gtex_mapper = {'Adipose - Subcutaneous':                       'GTEX-ADI', 
                        'Adipose - Visceral (Omentum)':                 'GTEX-ADI',
                        'Adrenal Gland':                                'GTEX-ADR_GLA',
                        'Artery - Aorta':                               'GTEX-BLO_VSL',
                        'Artery - Coronary':                            'GTEX-BLO_VSL',
                        'Artery - Tibial':                              'GTEX-BLO_VSL',
                        'Bladder':                                      'GTEX-BLA',
                        'Brain - Amygdala':                             'GTEX-BRA',
                        'Brain - Anterior cingulate cortex (BA24)':     'GTEX-BRA',
                        'Brain - Caudate (basal ganglia)':              'GTEX-BRA',
                        'Brain - Cerebellar Hemisphere':                'GTEX-BRA',
                        'Brain - Cerebellum':                           'GTEX-BRA',
                        'Brain - Cortex':                               'GTEX-BRA',
                        'Brain - Frontal Cortex (BA9)':                 'GTEX-BRA',
                        'Brain - Hippocampus':                          'GTEX-BRA',
                        'Brain - Hypothalamus':                         'GTEX-BRA',
                        'Brain - Nucleus accumbens (basal ganglia)':    'GTEX-BRA',
                        'Brain - Putamen (basal ganglia)':              'GTEX-BRA',
                        'Brain - Spinal cord (cervical c-1)':           'GTEX-BRA',
                        'Brain - Substantia nigra':                     'GTEX-BRA',
                        'Breast - Mammary Tissue':                      'GTEX-BRE',
                        'Cells - Cultured fibroblasts':                 'GTEX-SKI',
                        'Cells - EBV-transformed lymphocytes':          'GTEX-BLO',
                        'Cells - Leukemia cell line (CML)':             'TCGA-LAML', # TODO: Know what to do with this category. This is a problematic category because it is GTEX but appears to be of sick patients
                        'Cervix - Ectocervix':                          'GTEX-CER',
                        'Cervix - Endocervix':                          'GTEX-CER',
                        'Colon - Sigmoid':                              'GTEX-COL',
                        'Colon - Transverse':                           'GTEX-COL',
                        'Esophagus - Gastroesophageal Junction':        'GTEX-ESO',
                        'Esophagus - Mucosa':                           'GTEX-ESO',
                        'Esophagus - Muscularis':                       'GTEX-ESO',
                        'Fallopian Tube':                               'GTEX-FAL_TUB',
                        'Heart - Atrial Appendage':                     'GTEX-HEA',
                        'Heart - Left Ventricle':                       'GTEX-HEA',
                        'Kidney - Cortex':                              'GTEX-KID',
                        'Kidney - Medulla':                             'GTEX_KID',
                        'Liver':                                        'GTEX-LIV',
                        'Lung':                                         'GTEX-LUN',
                        'Minor Salivary Gland':                         'GTEX-SAL_GLA',
                        'Muscle - Skeletal':                            'GTEX-MUS',
                        'Nerve - Tibial':                               'GTEX-NER',
                        'Ovary':                                        'GTEX-OVA',
                        'Pancreas':                                     'GTEX-PAN',
                        'Pituitary':                                    'GTEX-PIT',
                        'Prostate':                                     'GTEX-PRO',
                        'Skin - Not Sun Exposed (Suprapubic)':          'GTEX-SKI',
                        'Skin - Sun Exposed (Lower leg)':               'GTEX-SKI',
                        'Small Intestine - Terminal Ileum':             'GTEX-SMA_INT',
                        'Spleen':                                       'GTEX-SPL',
                        'Stomach':                                      'GTEX-STO',
                        'Testis':                                       'GTEX-TES',
                        'Thyroid':                                      'GTEX-THY',
                        'Uterus':                                       'GTEX-UTE',
                        'Vagina':                                       'GTEX-VAG',
                        'Whole Blood':                                  'GTEX-BLO'
                    }

# There are a total of 57/740 problematic samples
recount3_normal_tcga_mapper = { 'Adrenal Gland':        'GTEX-ADR_GLA',
                                'Bile Duct':            'GTEX-LIV',     # Problematic 9 samples
                                'Bladder':              'GTEX-BLA',
                                'Brain':                'GTEX-BRA',
                                'Breast':               'GTEX-BRE',
                                'Cervix':               'GTEX-CER',
                                'Colorectal':           'GTEX-COL',
                                'Esophagus':            'GTEX-ESO',
                                'Head and Neck':        'GTEX-SKI',     # Problematic 44 samples (Squamous cells are found in the outer layer of skin and in the mucous membranes)
                                'Kidney':               'GTEX-KID',
                                'Liver':                'GTEX-LIV',
                                'Lung':                 'GTEX-LUN',
                                'Pancreas':             'GTEX-PAN',
                                'Prostate':             'GTEX-PRO',
                                'Skin':                 'GTEX-SKI',
                                'Soft Tissue':          'GTEX-ADI',     # Problematic 2 samples
                                'Stomach':              'GTEX-STO',
                                'Thymus':               'GTEX-SKI',     # Problematic 2 samples (Thymoma and thymic carcinoma, also called thymic epithelial tumors (TETs))
                                'Thyroid':              'GTEX-THY',
                                'Uterus':               'GTEX-UTE'
                                }


# Create mapper directory if there is not one
os.makedirs(os.path.join("data", "toil_data", "mappers"), exist_ok=True)
os.makedirs(os.path.join("data", "wang_data", "mappers"), exist_ok=True)
os.makedirs(os.path.join("data", "recount3_data", "mappers"), exist_ok=True)


# FIXME: The names of the files should be much more standar

# Save normal_tcga_mapper mappers to file
with open(os.path.join("data", "toil_data", "mappers", "normal_tcga_2_gtex_mapper.json"), 'w') as f:
    json.dump(normal_tcga_mapper, f, indent=4)
# Save phenotype mapper mappers to file
with open(os.path.join("data", "toil_data", "mappers", "phenotype_mapper.json"), 'w') as f:
    json.dump(phenotype_mapper, f, indent=4)
# Save category mapper to file
with open(os.path.join("data", "toil_data", "mappers", "category_mapper.json"), 'w') as f:
    json.dump(category_mapper, f, indent=4)
# Save id_2_tissue_mapper to file
with open(os.path.join("data", "toil_data", "mappers", "id_2_tissue_mapper.json"), 'w') as f:
    json.dump(id_2_tissue_mapper, f, indent=4)


# Save wang_standard mapper to file
with open(os.path.join("data", "wang_data", "mappers", "wang_standard_label_mapper.json"), 'w') as f:
    json.dump(wang_standard_label_mapper, f, indent=4)

# Save recount3_standard mapper to file
with open(os.path.join("data", "recount3_data", "mappers", "recount3_gtex_mapper.json"), 'w') as f:
    json.dump(recount3_gtex_mapper, f, indent=4)
with open(os.path.join("data", "recount3_data", "mappers", "healthy_tcga_2_gtex_mapper.json"), 'w') as f:
    json.dump(recount3_normal_tcga_mapper, f, indent=4)
