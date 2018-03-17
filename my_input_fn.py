def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """input data in tensorflow
    
    :type features: DataFrame
    :type targets: DataFrame
    :type batch_size: int
    :type shuffle: boolean
    :type num_epochs: int, None = repeat indefinitely
    :rtype (features, labels): Tuple
    """
    features = {key:np.array(value) for key,value in dict(features).items()}
    
    ds = Dataset.from_tensor_slices((features, targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    if shuffle:
        ds = ds.shuffle(10000)
        
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels