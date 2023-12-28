from bing_image_downloader import downloader


class DataCrawler:
    @staticmethod
    def downloadimages(query: str, output_dir: str, limit=10):
        """
        download images from Google
        :param query: google query to download from
        :param output_dir: where to store the downloaded data
        :param limit: number of entries to download as limit
        :return:
        """
        downloader.download(
            query,
            limit=limit,
            output_dir=output_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=60
        )
