import logging
from urllib import request
from zipfile import ZipFile

DATA_POP = "http://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GPW4_GLOBE_R2015A/GHS_POP_GPW42015_GLOBE_R2015A_54009_250/V1-0/GHS_POP_GPW42015_GLOBE_R2015A_54009_250_v1_0.zip"


logger = logging.getLogger(__name__)

def download(url, target_directory):
    logger.info(f"Downloading from {url} to {target_directory}.")
    request.urlretrieve(url, target_directory)


def unzip(zip_filepath, destination):
    with ZipFile(zip_filepath, "r") as zObject:
        zObject.extractall(path=destination)


if __name__ == "__main__":
    download(DATA_POP, "data/population.zip")
    unzip("data/population.zip", "data/")
    logger.info("Data downloaded and unzipped.")