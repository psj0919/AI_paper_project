def dataset_info(dataset_name='demo'):
    if dataset_name == "paper_dataset":
        train_path = 'C:/Users/82107/AI_project_dataset/data/Training/labeling_data/training_paper/training_논문/논문요약20231006_0.json'

    return train_path
def get_config_dict():

    dataset_name = "paper_dataset"
    train_path= dataset_info(dataset_name)

    dataset = dict(
        train_path = train_path,
        image_size = 224,
        batch_size = 16,
        num_workers = 5,
    )
    model = dict(
        name = 't5'
    )
    solver = dict(
        epoch=100,
        lr = 1e-4,
        lr_base= 1e-5,
        lr_max = 0.5e-7,
        lr_gamma = 0.9,
        lrf = 1e-2,
        T_up = 10,
        T_down = 10,
        weight_decay = 5e-4,
        print_freq = 100,
        eval_interval = 25000 * 4,
        num_thres = 120,
        num_up_down = 50,
    )
    option = dict(
        gpu_id='1',
        fc_metric = 'arc',
        easy_margin=False,
        optimizer = 'adam',
        scheduler = 'cosine'
    )
    config = dict(
        dataset = dataset,
        model = model,
        solver = solver,
        option = option
    )

    return config

