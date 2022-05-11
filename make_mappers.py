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
                    'Cells - Leukemia Cell Line (Cml)':             'TCGA-LAML', # TODO: Know wtf to do with this category. This is a problematic category because it is GTEX but appears to be of sick patients
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
# There are a total of 93/738 problematic categories
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

# Define a label mapper for all the values in category mapper
sorted_labels = sorted(category_mapper.values())
label_mapper = {value: i for i, value in enumerate(sorted_labels)}

# Create mapper directory if there is not one
if not os.path.exists(os.path.join("data", "toil_data", "mappers")):
    os.makedirs(os.path.join("data", "toil_data", "mappers"))

# Save normal_tcga_mapper mappers to file
with open(os.path.join("data", "toil_data", "mappers", "normal_tcga_2_gtex_mapper.json"), 'w') as f:
    json.dump(normal_tcga_mapper, f, indent=4)
# Save phenotype mapper mappers to file
with open(os.path.join("data", "toil_data", "mappers", "phenotype_mapper.json"), 'w') as f:
    json.dump(phenotype_mapper, f, indent=4)
# Save category mapper to file
with open(os.path.join("data", "toil_data", "mappers", "category_mapper.json"), 'w') as f:
    json.dump(category_mapper, f, indent=4)
# Save label mapper to file
with open(os.path.join("data", "toil_data", "mappers", "lab_txt_2_lab_num_mapper.json"), 'w') as f:
    json.dump(label_mapper, f, indent=4)