from PIL import Image, ImageChops
from tools.path_settings import EXPER_PATH

ORIG_DIR = "PerfectPredictions"
REPLI_DIR = "20210120"
OUTPUT_DIR = '/Subtractions/' + REPLI_DIR + '-' + ORIG_DIR + '_'
images = ['00058_072', '00179_873', '00496_726', '00548_211', '00806_568',
          '00935_077', '01113_119', '01158_776', '01187_674']


if __name__ == "__main__":
    for image in images:
        orig_img = Image.open(EXPER_PATH + ORIG_DIR + '/' + image + '_orig.png').convert('L')
        repli_img = Image.open(EXPER_PATH + REPLI_DIR + '/' + image + '_orig.png').convert('L')
        subtracted_img = ImageChops.subtract(repli_img, orig_img, scale=1.0)
        subtracted_img.save(EXPER_PATH + OUTPUT_DIR + image + '.png')
