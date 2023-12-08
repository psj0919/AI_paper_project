def dataset_info(dataset_name):

    if dataset_name == 'data_1':
        train_path = '/home/work/paper_data/논문요약20231006_0.json'

    return dataset_name, train_path 

def get_config_dict():
    dataset_name = 'data_1'
    name, data = dataset_info(dataset_name)
    checkpoints_save_path = "./checkpoints/"
    checkpoints_file = None # or my checkpoint_path

    dataset = dict(
        name = name,
        data_path = data, 
        checkpoints_file = checkpoints_file,
        checkpoints_save_path = checkpoints_save_path,
        max_size = 5000,
        embedding_dim = 100,
        hidden_dim = 256,
        output_dim = 1,
        n_layers = 2,
        bidirectional = True,
        dropout = 0.5
    )
    model = dict(
        name = 'LSTM',

    )
    solver = dict(
        batch_size = 64,
        epoch = 100
        
    )
    option = dict(
        loss = 'BCE',
        optim = 'Adam',
        
    )
    config = dict(
        dataset = dataset,
        model = model,
        solver = solver,
        option = option
    )

    return config

