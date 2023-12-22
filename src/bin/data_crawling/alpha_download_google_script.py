import argparse

from src.data.crawling import DataCrawler


def create_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--dataset_path',
        metavar='dataset_path',
        type=str,
        help='path to where download the crawled data'
    )
    args = parser.parse_args()
    return args


def main():
    args = create_args()
    DataCrawler.downloadimages('trout caught')


if __name__ == '__main__':
    main()
