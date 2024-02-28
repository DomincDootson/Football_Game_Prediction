import pandas as pd

def get_upsampled_data(X, y): # Note that this will return quite a large array, it might need to be down sampled afterwards to make the linear response fit quicker 
    data = y.join(X).copy()
    target_feature = y.columns[0]
    class_counts = data[target_feature].value_counts()
    
    # Calculate the maximum count among all classes
    max_count = class_counts.max()
    
    # Upsample each class to match the count of the class with the maximum count
    upsampled_data = pd.DataFrame(columns = data.columns)
    for class_label, count in class_counts.items():
        class_data = data[data[target_feature] == class_label]

        if count < max_count:
            upsampled_class_data = class_data.sample(n=max_count - count, replace=True, random_state=42)            
            upsampled_data = pd.concat([upsampled_data, upsampled_class_data, class_data])
        else:
            upsampled_data = pd.concat([upsampled_data, class_data])
    
    # Shuffle the upsampled data
    upsampled_data = upsampled_data.sample(frac=1, random_state=42).reset_index(drop=True).copy()
    return upsampled_data


def get_fraction_of_data(data, n_each_class, target_feature):
    class_counts = data[target_feature].value_counts()
    down_sampled_data = pd.DataFrame(columns = data.columns)

    for class_label, _ in class_counts.items():
        class_data = data[data[target_feature] == class_label]
        should_replace = True if len(class_data) < n_each_class else False # Do this to make sure that we can get then right amount
        down_sampled_class_data = class_data.sample(n=n_each_class, replace=should_replace, random_state=42)            
        down_sampled_data = pd.concat([down_sampled_data, down_sampled_class_data])

    return down_sampled_data