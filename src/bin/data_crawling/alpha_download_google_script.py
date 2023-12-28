import argparse
import os

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
    queries = [
        ('trout', 'trout caught'),
        ('catfish', 'catfish caught'),
        ('pike', 'pike caught'),
        ('zander', 'zander caught'),
        ('perch', 'perch caught'),
        ('carp', 'carp caught'),
        ('barbel', 'barbel caught'),
        ('eel', 'eel caught'),
        ('cod', 'cod caught'),
        ('mackerel', 'mackerel caught'),
    ]

    for name, q in queries:
        DataCrawler.downloadimages(
            q,
            output_dir=os.path.join(args.dataset_path, name),
            limit=100,
        )


if __name__ == '__main__':
    main()
