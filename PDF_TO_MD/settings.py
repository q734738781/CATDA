ocr_settings = {'use_gpu': True,
                'debug' : False,
                'db_unclip_ratio': 4,
                'det_db_box_thresh': 0.3,
                'use_dilation': True,
                'gpu_mem': 8192,
                'lang':'en',
                'use_space_char':True,
                'show_log': False,
                'rec_model_dir' : r'../PaddleOCR/Models/en_PP-OCRv4_rec_infer',
                'det_model_dir' : r'../PaddleOCR/Models/en_PP-OCRv3_det_infer',
                'table_model_dir' : r'../PaddleOCR/Models/en_ppstructure_mobile_v2.0_SLANet_infer',
                'layout_model_dir': r'../PaddleOCR/Models/picodet_lcnet_x1_0_fgd_layout_infer',
}
zoom_factor = 8
debug_result = False
azure_api_key,azure_api_endpoint = "API_KEY_HERE","ENDPOINT_HERE"
azure_auto_load = True
azure_output_dpi = 300

section_names = ["Abstract",
                            'Introduction', 'Related Work', 'Background',
                            "Preliminary", "Problem Formulation",
                            'Methods', 'Methodology',
                            "Materials and Methods", "Experiment Settings", "Experimental Results", "Evaluation", "Experiments",
                            "Results", 'Findings', 'Data Analysis',
                            "Discussion", "Results and Discussion", "Conclusion",
                            'References', 'Acknowledgements'] + [
                               "Literature Review",
                               "Theoretical Framework",
                               "Experimental Design",
                               "Data Collection",
                               "Data Collection Methods",
                               "Data Sources",
                               "Study Area",
                               "Research Design",
                               "Research Methodology",
                               "Statistical Analysis",
                               "Results Overview",
                               "Model Development",
                               "Discussion of Results",
                               "Interpretation",
                               "Implications",
                               "Limitations",
                               "Future Work",
                               "Recommendations",
                               "Concluding Remarks",
                               "Appendix",
                               "Supplementary Materials",
                               "Ethical Considerations",
                               "Funding",
                               "Author Contributions",
                               "Conflict of Interest",
                               "References and Notes",
                               "Citations",
                               "Bibliography"
                           ]

first_upper_section_names = [name[0].upper() + name[1:].lower() for name in section_names]
full_upper_section_names = [name.upper() for name in section_names]
predefined_section_names = section_names + first_upper_section_names + full_upper_section_names

substitute_special_char = {"ﬂ":'fl',
                                  "ﬁ":"fi"}
