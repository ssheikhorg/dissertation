from datasets import load_dataset


def load_benchmark(name):
    dataset_map = {
        'truthful_qa': 'truthful_qa',
        'hellaswag': 'hellaswag',
        'winobias': ('winobias', 'coreference')
    }

    if isinstance(dataset_map[name], tuple):
        return load_dataset(*dataset_map[name])
    return load_dataset(dataset_map[name])
