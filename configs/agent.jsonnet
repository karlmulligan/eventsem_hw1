{
    dataset_reader: {
        type: "uds_srl_reader", 
        lazy: false
    },
    train_data_path: "data/agent/train.json",
    validation_data_path: "data/agent/dev.json",
    data_loader: {
        batch_sampler: {
            type: "bucket",
            batch_size: 10
        }
    },
    model: {
        type: "srl_lstm",
        embedder: {
            token_embedders: {
                tokens: {
                    type: "embedding",
                          pretrained_file: "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
                          embedding_dim: 50,
                          trainable: false
                }
            }
        },
        encoder: {
            type: "lstm",
                  input_size: 50,
                  hidden_size: 25,
                  bidirectional: true
        }
    },
    trainer: {
        num_epochs: 10,
        patience: 3,
        cuda_device: 0,
        grad_clipping: 5.0,
        validation_metric: '-loss',
        optimizer: {
            type: 'adam',
            lr: 0.003
        }
    }
}
