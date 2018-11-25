import numpy as np
import urllib, cStringIO

from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from scipy.misc import imresize

from PIL import Image


class Augments():

	def __init__(self):

		self.AUG_BATCH_SIZE = 2
		self.REFERENCE_MU_LAB = [8.63234435, -0.11501964, 0.03868433]
		self.REFERENCE_STD_LAB = [0.57506023, 0.10403329, 0.01364062]

		self.IMAGE_WIDTH = 224
		self.IMAGE_HEIGHT = 224
		self.IMAGE_CHANS = 3
		self.IMAGE_DTYPE = "float32"
		self.IMAGE_SHAPE = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3)

		# define conversion matrices
		self._rgb2lms = np.array([[0.3811, 0.5783, 0.0402],
							 [0.1967, 0.7244, 0.0782],
							 [0.0241, 0.1288, 0.8444]])

		self._lms2lab = np.dot(
			np.array([[1 / (3 ** 0.5), 0, 0],
					  [0, 1 / (6 ** 0.5), 0],
					  [0, 0, 1 / (2 ** 0.5)]]),
			np.array([[1, 1, 1],
					  [1, 1, -2],
					  [1, -1, 0]])
		)

		# Define conversion matrices
		self._lms2rgb = np.linalg.inv(self._rgb2lms)
		self._lab2lms = np.linalg.inv(self._lms2lab)

	def rgb_to_lab(self, im_rgb):
		# get input image dimensions
		m, n, c = im_rgb.shape
		# calculate im_lms values from RGB
		im_rgb = np.reshape(im_rgb, (m * n, 3))
		im_lms = np.dot(self._rgb2lms, np.transpose(im_rgb))
		im_lms[im_lms == 0] = np.spacing(1)
		# calculate LAB values from im_lms
		im_lab = np.dot(self._lms2lab, np.log(im_lms))
		# reshape to 3-channel image
		im_lab = np.reshape(im_lab.transpose(), (m, n, 3))
		return im_lab

	def lab_to_rgb(self, im_lab):
		# get input image dimensions
		m, n, c = im_lab.shape
		# Define conversion matrices
		self._lms2rgb = np.linalg.inv(self._rgb2lms)
		self._lab2lms = np.linalg.inv(self._lms2lab)
		# calculate im_lms values from LAB
		im_lab = np.reshape(im_lab, (m * n, 3))
		im_lms = np.dot(self._lab2lms, np.transpose(im_lab))
		# calculate RGB values from im_lms
		im_lms = np.exp(im_lms)
		im_lms[im_lms == np.spacing(1)] = 0
		im_rgb = np.dot(self._lms2rgb, im_lms)
		# reshape to 3-channel image
		im_rgb = np.reshape(im_rgb.transpose(), (m, n, 3))
		return im_rgb

	def reinhard(self, im_src, t_mu, t_sigma, s_mu, s_sigma):
		m, n, c = im_src.shape
		# convert input image to LAB color space
		im_lab = self.rgb_to_lab(im_src)
		# calculate s_mu if not provided
		if s_mu is None:
			s_mu = im_lab.sum(axis=0).sum(axis=0) / (m * n)
		# center to zero-mean
		for i in range(3):
			im_lab[:, :, i] = im_lab[:, :, i] - s_mu[i]
		# calculate s_sigma if not provided
		if s_sigma is None:
			s_sigma = ((im_lab * im_lab).sum(axis=0).sum(axis=0) /
					   (m * n - 1)) ** 0.5
		# scale to unit variance
		for i in range(3):
			im_lab[:, :, i] = im_lab[:, :, i] / s_sigma[i]
		# rescale and recenter to match target statistics
		for i in range(3):
			im_lab[:, :, i] = im_lab[:, :, i] * t_sigma[i] + t_mu[i]
		# convert back to RGB colorspace
		im_normalized = self.lab_to_rgb(im_lab)
		im_normalized[im_normalized > 255] = 255
		im_normalized[im_normalized < 0] = 0
		im_normalized = im_normalized.astype(np.uint8)
		return im_normalized

	def prepare_image(self, aurl, slide):

		img = np.array(Image.open(
			cStringIO.StringIO(urllib.urlopen(aurl).read())
			))

		wsi_mean_std = self.find_mean_std(slide)

		img_norm = self.reinhard(img, self.REFERENCE_MU_LAB, self.REFERENCE_STD_LAB, wsi_mean_std[0], wsi_mean_std[1])

		img_norm = img_to_array(
			imresize(img_norm, self.IMAGE_SHAPE)
			)

		image_dim = np.expand_dims(img_norm, axis=0)
		batch_angle = self.generator(image_dim, rotation=60)
		batch_angle = batch_angle.reshape(2, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3)
		batches = np.round(batch_angle).astype(np.uint8)
		image = preprocess_input(batches)
		return image

	def generator(self, img, rotation=0., preprocess_fcn=None):
		datagen = ImageDataGenerator(
			rotation_range=rotation,
			fill_mode='nearest',
			preprocessing_function=preprocess_fcn,
			data_format=K.image_data_format())
		datagen.fit(img)
		index = 0
		batch_img = []
		for img_batch in datagen.flow(img, batch_size=2, shuffle=False):
			for img in img_batch:
				batch_img = img if index == 0 else np.append(batch_img, img, axis=0)
				index += 1
			if index >= self.AUG_BATCH_SIZE:
				break
		return batch_img

	def find_mean_std(self, slide):
		return {
			'TCGA-BH-A0WA-01Z-00-DX1': [[8.12304011, -0.21879297, 0.03182605], [0.94722531, 0.20476486, 0.02773344]],
			'TCGA-E2-A1LI-01Z-00-DX1': [[8.86278778, -0.03419073, 0.03992084], [0.54908458, 0.09393901, 0.03210528]],
			'TCGA-AR-A0U1-01Z-00-DX1': [[8.54579346, -0.08465618, 0.05318259], [0.5631746, 0.10049048, 0.02898772]],
			'TCGA-C8-A27B-01Z-00-DX1': [[7.70377597, -0.2009137, 0.09460135], [0.88591483, 0.18358379, 0.04669634]],
			'TCGA-AQ-A04J-01Z-00-DX1': [[8.39437157, -0.13627366, 0.04264897], [0.58746401, 0.11733428, 0.02470397]],
			'TCGA-C8-A3M7-01Z-00-DX1': [[9.02011267, -0.04565693, 0.0274629], [0.43183584, 0.08530048, 0.02019638]],
			'TCGA-AN-A0AT-01Z-00-DX1': [[8.26446559, -0.15933994, 0.05822676], [0.73578885, 0.13794817, 0.03208397]],
			'TCGA-AR-A0TS-01Z-00-DX1': [[8.43862243, -0.08165163, 0.06870721], [0.54924008, 0.11867078, 0.03121097]],
			'TCGA-A7-A6VY-01Z-00-DX1': [[8.22309827, -0.16774565, 0.05891794], [0.59324408, 0.11831668, 0.02453208]],
			'TCGA-AR-A256-01Z-00-DX1': [[8.63989151, -0.04862964, 0.05356381], [0.48542311, 0.08193346, 0.02741553]],
			'TCGA-BH-A1F6-01Z-00-DX1': [[8.84938551, -0.02846924, 0.04545655], [0.32182043, 0.05792674, 0.0219732, ]],
			'TCGA-D8-A27F-01Z-00-DX1': [[8.95546947, -0.06551688, 0.02160668], [0.51702638, 0.09148701, 0.01661208]],
			'TCGA-AC-A6IW-01Z-00-DX1': [[7.6510243, -0.28666384, 0.06082674], [0.79260214, 0.16323221, 0.03004769]],
			'TCGA-AR-A2LR-01Z-00-DX1': [[8.83645072, -0.04384143, 0.03867131], [0.47718551, 0.0666339, 0.0285149]],
			'TCGA-BH-A0BL-01Z-00-DX1': [[8.39489421, -0.07597604, 0.04233761], [0.85768505, 0.13484808, 0.03577936]],
			'TCGA-A1-A0SK-01Z-00-DX1': [[8.86949325, -0.10216824, 0.01805525], [0.52531549, 0.10402657, 0.02806813]],
			'TCGA-AC-A2QJ-01Z-00-DX1': [[8.77326124, -0.10480836, 0.03948286], [0.38484823, 0.07641309, 0.0177293, ]],
			'TCGA-AN-A0G0-01Z-00-DX1': [[8.3012587, -0.16648116, 0.06235426], [0.6413291, 0.12723023, 0.02884364]],
			'TCGA-D8-A13Z-01Z-00-DX1': [[9.16297503, -0.03268581, 0.01784125], [0.30000155, 0.05603581, 0.01225873]],
			'TCGA-E2-A14N-01Z-00-DX1': [[8.31673226, -0.14909013, 0.06712033], [0.66477118, 0.12513515, 0.03444409]],
			'TCGA-A2-A0CM-01Z-00-DX1': [[8.15721997, -0.26266193, 0.06758453], [0.65695962, 0.15945036, 0.03702212]],
			'TCGA-C8-A1HJ-01Z-00-DX1': [[8.56262533, -0.14475003, 0.03722465], [0.76917447, 0.16636545, 0.02959971]],
			'TCGA-D8-A1XQ-01Z-00-DX1': [[8.60967583, -0.10439691, 0.04227104], [0.71322219, 0.12718897, 0.02671343]],
			'TCGA-D8-A143-01Z-00-DX1': [[8.9322165, -0.06512799, 0.02179042], [0.57870641, 0.10096768, 0.0186204]],
			'TCGA-A7-A26I-01Z-00-DX1': [[8.2426081, -0.17309903, 0.04425223], [0.62441161, 0.12262431, 0.02420914]],
			'TCGA-AR-A1AR-01Z-00-DX1': [[8.19425035, -0.11979294, 0.07410053], [0.73081281, 0.1456065, 0.0332639, ]],
			'TCGA-AN-A0XU-01Z-00-DX1': [[8.19002864, -0.17651642, 0.05922504], [0.91571089, 0.19031635, 0.03963962]],
			'TCGA-AC-A2QH-01Z-00-DX1': [[8.85950139, -0.08432725, 0.03520345], [0.33763828, 0.06627839, 0.01801243]],
			'TCGA-A1-A0SP-01Z-00-DX1': [[8.62887071, -0.08999701, 0.04568819], [0.57904781, 0.11951785, 0.0252132, ]],
			'TCGA-BH-A18G-01Z-00-DX1': [[8.09391686, -0.24677524, 0.01648765], [0.72579067, 0.15928352, 0.02182084]],
			'TCGA-3C-AALJ-01Z-00-DX1': [[8.84972554, -0.07709956, 0.05975043], [0.35611718, 0.05796765, 0.02562233]],
			'TCGA-E2-A150-01Z-00-DX1': [[8.46530408, -0.0905464, 0.06944721], [0.62584951, 0.12133655, 0.04026832]],
			'TCGA-A2-A04T-01Z-00-DX1': [[8.52935559, -0.17421487, 0.0495899], [0.57290032, 0.12003877, 0.03093144]],
			'TCGA-A2-A04T-01Z-00-DX1': [[8.5318386, -0.17373998, 0.04945857], [0.57388937, 0.12015781, 0.03097241]],
			'TCGA-BH-A1FC-01Z-00-DX1': [[9.11509148, -0.0627013, 0.011339, ], [0.45187575, 0.09249365, 0.0141376, ]],
			'TCGA-EW-A1P7-01Z-00-DX1': [[8.88951271, -0.06362235, 0.03132687], [0.56383634, 0.10274191, 0.02626814]],
			'TCGA-E2-A1LK-01Z-00-DX1': [[8.88361746, -0.04486375, 0.0396937, ], [0.38883251, 0.07170387, 0.02346176]],
			'TCGA-AC-A2QH-01Z-00-DX1': [[8.86157476, -0.08400257, 0.03510868], [0.33717649, 0.06609482, 0.01803808]],
			'TCGA-BH-A0E6-01Z-00-DX1': [[8.65540344, -0.08593961, 0.04038695], [0.71035827, 0.11542384, 0.03284969]],
			'TCGA-A7-A5ZV-01Z-00-DX1': [[8.13269312, -0.12858156, 0.07244753], [0.54656459, 0.11445225, 0.03418238]],
			'TCGA-AR-A0U1-01Z-00-DX1': [[8.54576705, -0.08466658, 0.05318058], [0.56330636, 0.10050811, 0.02898489]],
			'TCGA-E2-A1LI-01Z-00-DX1': [[8.86280417, -0.0341883, 0.0399185], [0.54910873, 0.09393349, 0.03210572]],
			'TCGA-AR-A256-01Z-00-DX1': [[8.6385431, -0.04870856, 0.053672], [0.48429063, 0.08191319, 0.02733405]],
			'TCGA-AN-A0G0-01Z-00-DX1': [[8.30508317, -0.16591354, 0.06218972], [0.64185865, 0.12716973, 0.02890377]],
			'TCGA-D8-A143-01Z-00-DX1': [[8.93225615, -0.06511906, 0.02179006], [0.57868555, 0.10095854, 0.01861954]],
			'TCGA-E2-A14R-01Z-00-DX1': [[8.8689715, -0.07349128, 0.03293157], [0.59941287, 0.12313357, 0.03359943]],
			'TCGA-AN-A0AT-01Z-00-DX1': [[8.26984706, -0.15861369, 0.05798365], [0.73708563, 0.13787429, 0.03214885]],
			'TCGA-C8-A1HJ-01Z-00-DX1': [[8.5570678, -0.1456806, 0.03740457], [0.76964106, 0.16666946, 0.02960932]],
			'TCGA-E2-A1B6-01Z-00-DX1': [[8.86878124, -0.08162481, 0.02952742], [0.55003118, 0.10604912, 0.02114139]],
			'TCGA-AC-A6IW-01Z-00-DX1': [[7.65451444, -0.28594901, 0.06072497], [0.79416539, 0.16337285, 0.03007741]],
			'TCGA-BH-A1F6-01Z-00-DX1': [[8.85299086, -0.02801334, 0.04528508], [0.3204699, 0.0573719, 0.02205534]],
			'TCGA-A2-A04P-01Z-00-DX1': [[8.59006181, -0.16536107, 0.05174068], [0.55775879, 0.11463206, 0.03229714]],
			'TCGA-3C-AALJ-01Z-00-DX1': [[8.84973351, -0.0769805, 0.05984422], [0.35446575, 0.05757344, 0.02549627]],
			'TCGA-BH-A18G-01Z-00-DX1': [[8.09595526, -0.2463604, 0.01647693], [0.72699977, 0.15938887, 0.02180937]],
			'TCGA-AR-A0TS-01Z-00-DX1': [[8.43862731, -0.08165407, 0.06870567], [0.54927632, 0.11867864, 0.03121179]],
			'TCGA-A2-A3XS-01Z-00-DX1': [[8.87856927, -0.09872476, 0.04610452], [0.48373801, 0.08422579, 0.02979942]],
			'TCGA-A7-A26I-01Z-00-DX1': [[8.24275673, -0.17307844, 0.04425084], [0.62439958, 0.12262561, 0.02421649]],
			'TCGA-AQ-A54N-01Z-00-DX1': [[8.3334363, -0.15156639, 0.04509003], [0.44230943, 0.09346407, 0.01692279]],
			'TCGA-D8-A13Z-01Z-00-DX1': [[9.16295938, -0.03268451, 0.01784217], [0.3000064, 0.05604103, 0.01225779]],
			'TCGA-D8-A1XQ-01Z-00-DX1': [[8.60968572, -0.10440199, 0.04227058], [0.71328394, 0.12720609, 0.02671668]],
			'TCGA-A7-A6VY-01Z-00-DX1': [[8.21841299, -0.16839632, 0.05912186], [0.59191497, 0.11836922, 0.02444649]],
			'TCGA-BH-A0BL-01Z-00-DX1': [[8.42562251, -0.07346505, 0.04110558], [0.85848285, 0.13320383, 0.03576917]],
			'TCGA-A2-A0CM-01Z-00-DX1': [[8.15502879, -0.26314849, 0.06768165], [0.6557993, 0.15928223, 0.03698183]],
			'TCGA-C8-A3M7-01Z-00-DX1': [[9.02013084, -0.04565214, 0.02746284], [0.43174292, 0.08528998, 0.02019785]],
			'TCGA-AR-A2LR-01Z-00-DX1': [[8.83647045, -0.04384123, 0.03867084], [0.47718488, 0.0666362, 0.02851578]]
		}.get(slide, [[8.12304011, -0.21879297, 0.03182605], [0.94722531, 0.20476486, 0.02773344]])
