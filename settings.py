# preprocessing
base_luna16_dir =  "E:/LUNA16/"
base_data_dir   = "E:/luna16_2021/"
annotation_file = base_luna16_dir + 'BMnodule.csv' # LIDCとLUNA16を対応させることによって生成されている
candidate_file  = base_luna16_dir + 'candidates_V2.csv' # LUNA16より
slice_png_benign_dir    = base_data_dir + 'benign_slice/'
slice_png_malignant_dir = base_data_dir + 'malignant_slice/'
slice_npy_benign_dir    = base_data_dir + 'benign_slice_npy/'
slice_npy_malignant_dir = base_data_dir + 'malignant_slice_npy/'

# netwirks
base_network_dir = "./networks/"
save_init_weight_dir = base_network_dir + 'init_weight/'
model_map_dir        = base_network_dir + 'model_map/'

# train
base_train_result_dir = "./result/train/"
save_model_dir  = base_train_result_dir + 'save_model/'
result_accs_dir = base_train_result_dir + 'accs/'
result_mets_dir = base_train_result_dir + 'mets/'
result_roc_dir  = base_train_result_dir + 'roc/'
result_doc_dir  = base_train_result_dir + 'doc/'

# visualize
visualize_sample_malignant_numbers = [
    # spicla
    "0_25_388",
    "0_27_421",
    "4_75_933",
    # notch
    "0_69_951",
    "1_28_280",
    "2_60_700",
    # part-solid
    "1_45_552",
    "2_0_4",
    "2_5_72",
    # cave
    "3_51_666",
    "3_67_853",
    "7_28_425",
    # cave
    "4_88_1084",
    "7_47_630",
    "9_51_641",
]

visualize_sample_benign_numbers = [
    # clear-boarder
    "0_14_135",
    "1_45_550",
    "8_27_424",
    # part-solid
    "1_50_657",
    "3_8_124",
    "5_8_60",
    # solid
    "0_82_1046",
    "1_79_1028",
    "2_71_925",
]

base_visualize_result_dir = './result/visualize/'
result_heatmaps_npy_dir = base_visualize_result_dir + 'heatmaps_npy/'
result_heatmaps_png_dir = base_visualize_result_dir + 'heatmaps_png/'
result_heatmaps_experiments_vs_layers_dir = base_visualize_result_dir + 'heatmaps_layers_vs_experiment/'
result_heatmaps_groundtruth_npy_dir = base_visualize_result_dir + 'heatmaps_groundtruth_npy/'
result_heatmaps_groundtruth_png_dir = base_visualize_result_dir + 'heatmaps_groundtruth_png/'

# compare
base_compare_result_dir = './result/compare/'
result_compare_dice_m_csv = base_compare_result_dir + 'dice_m.csv'
result_compare_dice_b_csv = base_compare_result_dir + 'dice_b.csv'
result_compare_jaccard_m_csv = base_compare_result_dir + 'jaccard_m.csv'
result_compare_jaccard_b_csv = base_compare_result_dir + 'jaccard_b.csv'
result_compare_simpson_m_csv = base_compare_result_dir + 'simpson_m.csv'
result_compare_simpson_b_csv = base_compare_result_dir + 'simpson_b.csv'
base_groundtruth_dir = './groundtruth/'
doctor_csv = base_groundtruth_dir + 'doctor.csv'
