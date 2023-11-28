from src.fish_classification.data import DataUtils


def main():
    dataset_path = r'C:\data\fish_judge\fish_data\Fish_Dataset\Fish_Dataset'
    DataUtils.create_generators(dataset_path)
    exit(0)


if __name__ == '__main__':
    main()
