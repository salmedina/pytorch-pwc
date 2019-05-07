#!/usr/bin/env python

import torch

import argparse
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
from glob import glob
import os.path as osp
import time
import pandas as pd

try:
	from correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 40) # requires at least pytorch version 0.4.0
torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.cuda.device(1) # change this if you have a multiple graphics cards and you want to utilize them
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance


##########################################################

Backward_tensorGrid = {}
Backward_tensorPartial = {}

def Backward(tensorInput, tensorFlow):
	if str(tensorFlow.size()) not in Backward_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

		Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
	# end

	if str(tensorFlow.size()) not in Backward_tensorPartial:
		Backward_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones([ tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3) ])
	# end

	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
	tensorInput = torch.cat([ tensorInput, Backward_tensorPartial[str(tensorFlow.size())] ], 1)

	tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

	tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

	return tensorOutput[:, :-1, :, :] * tensorMask

##########################################################

class Network(torch.nn.Module):
	def __init__(self, model_name):
		super(Network, self).__init__()

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tensorInput):
				tensorOne = self.moduleOne(tensorInput)
				tensorTwo = self.moduleTwo(tensorOne)
				tensorThr = self.moduleThr(tensorTwo)
				tensorFou = self.moduleFou(tensorThr)
				tensorFiv = self.moduleFiv(tensorFou)
				tensorSix = self.moduleSix(tensorFiv)

				return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

				if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.dblBackward = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
				)

			def forward(self, tensorFirst, tensorSecond, objectPrevious):
				tensorFlow = None
				tensorFeat = None

				if objectPrevious is None:
					tensorFlow = None
					tensorFeat = None

					tensorVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=tensorSecond), negative_slope=0.1, inplace=False)

					tensorFeat = torch.cat([ tensorVolume ], 1)

				elif objectPrevious is not None:
					tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
					tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

					tensorVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward)), negative_slope=0.1, inplace=False)

					tensorFeat = torch.cat([ tensorVolume, tensorFirst, tensorFlow, tensorFeat], 1)

				# end

				tensorFeat = torch.cat([ self.moduleOne(tensorFeat), tensorFeat], 1)
				tensorFeat = torch.cat([ self.moduleTwo(tensorFeat), tensorFeat], 1)
				tensorFeat = torch.cat([ self.moduleThr(tensorFeat), tensorFeat], 1)
				tensorFeat = torch.cat([ self.moduleFou(tensorFeat), tensorFeat], 1)
				tensorFeat = torch.cat([ self.moduleFiv(tensorFeat), tensorFeat], 1)

				tensorFlow = self.moduleSix(tensorFeat)

				return {
					'tensorFlow': tensorFlow,
					'tensorFeat': tensorFeat
				}

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()

				self.moduleMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)

			def forward(self, tensorInput):
				return self.moduleMain(tensorInput)

		self.moduleExtractor = Extractor()

		self.moduleTwo = Decoder(2)
		self.moduleThr = Decoder(3)
		self.moduleFou = Decoder(4)
		self.moduleFiv = Decoder(5)
		self.moduleSix = Decoder(6)

		self.moduleRefiner = Refiner()

		self.load_state_dict(torch.load('./network-' + model_name + '.pytorch'))

	def forward(self, tensorFirst, tensorSecond):
		tensorFirst = self.moduleExtractor(tensorFirst)
		tensorSecond = self.moduleExtractor(tensorSecond)

		objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
		objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
		objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
		objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
		objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)

		return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])

##########################################################

def estimate(moduleNetwork, tensorFirst, tensorSecond):
	assert(tensorFirst.size(2) == tensorSecond.size(2))
	assert(tensorFirst.size(3) == tensorSecond.size(3))

	tensorOutput = torch.FloatTensor()

	intWidth = tensorFirst.size(3)
	intHeight = tensorFirst.size(2)

	# There is no guarantee for correctness if the input size is not the same when training the model
	# comment this line out if you acknowledge this and want to continue
	#assert(intWidth == 1024)
	#assert(intHeight == 436)

	tensorFirst = tensorFirst.cuda()
	tensorSecond = tensorSecond.cuda()
	tensorOutput = tensorOutput.cuda()

	tensorPreprocessedFirst = tensorFirst
	tensorPreprocessedSecond = tensorSecond

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

	tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tensorFlow = 20.0 * torch.nn.functional.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	tensorOutput = tensorFlow.cpu()

	return tensorOutput
##########################################################

def profile_dir_processing(strModel, strInputDir, strFramesExt, strOutputDir):
	total_start_time = time.time()
	moduleNetwork = Network(strModel).cuda().eval()
	listFramesPath = sorted(glob(osp.join(strInputDir, '*%s'%strFramesExt)))

	processing_start_time = time.time()
	for intIndex, strFirst, strSecond in zip(range(len(listFramesPath)-1), listFramesPath[:-1], listFramesPath[1:]):
		tensorFirst = torch.FloatTensor(
			numpy.array(PIL.Image.open(strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
						1.0 / 255.0))
		tensorSecond = torch.FloatTensor(
			numpy.array(PIL.Image.open(strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
						1.0 / 255.0))

		tensorOutput = estimate(moduleNetwork, tensorFirst, tensorSecond)

		fileOutput = open(osp.join(strOutputDir, '%05d.flo'%(intIndex)), 'wb')

		numpy.array([80, 73, 69, 72], numpy.uint8).tofile(fileOutput)
		numpy.array([tensorOutput.size(2), tensorOutput.size(1)], numpy.int32).tofile(fileOutput)
		numpy.array(tensorOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(fileOutput)

		fileOutput.close()

	return time.time() - total_start_time, time.time() - processing_start_time

def load_rgb(rgb_path):
	return numpy.array(PIL.Image.open(rgb_path))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)

def process_dir(moduleNetwork, strInputDir, strOutputDir, strFramesExt='.jpg', bound=20):
	os.makedirs(strOutputDir, exist_ok=True)

	proc_batch_size = 100
	listAllFramesPath = sorted(glob(osp.join(strInputDir, '*%s' % strFramesExt)))
	total_frames = len(listAllFramesPath)

	for frame_index in range(0, total_frames, proc_batch_size):
		listFramesPath = listAllFramesPath[frame_index:frame_index+proc_batch_size]

		total_of = len(listFramesPath)-1

		template_img = load_rgb(listFramesPath[0])
		c, h, w = template_img.shape
		first_batch = numpy.zeros((total_of, c, h, w), dtype=numpy.float32)
		second_batch = numpy.zeros((total_of, c, h, w), dtype=numpy.float32)
		for intIndex, strFirst, strSecond in zip(range(total_of), listFramesPath[:-1], listFramesPath[1:]):
			first_batch[intIndex, ...] = load_rgb(strFirst)
			second_batch[intIndex, ...] = load_rgb(strSecond)

		tensorFirst = torch.FloatTensor(first_batch).cuda()
		tensorSecond = torch.FloatTensor(second_batch).cuda()

		tensorOutput = estimate(moduleNetwork, tensorFirst, tensorSecond)

		for intIndex in range(tensorOutput.shape[0]):
			tensorSave = tensorOutput[intIndex, ...]
			arrayOutput = numpy.array(tensorSave.numpy().transpose(1, 2, 0), numpy.float32)

			arrayOutput = ((arrayOutput + bound) / (2 * bound)) * 255.
			arrayOutput[arrayOutput < 0.] = 0.
			arrayOutput[arrayOutput > 255.] = 255.
			flow_x, flow_y = arrayOutput[..., 0], arrayOutput[..., 1]
			PIL.Image.fromarray(flow_x.astype(numpy.uint8), mode='L').save(
				osp.join(strOutputDir, 'flow_x_%05d.jpg' % (intIndex+frame_index)))
			PIL.Image.fromarray(flow_y.astype(numpy.uint8), mode='L').save(
				osp.join(strOutputDir, 'flow_y_%05d.jpg' % (intIndex+frame_index)))


def process_frames_dir(strModel, strFramesDir, strFlowDir, strFramesExt):
	listSourceDirs = sorted(glob(osp.join(strFramesDir, '*/')))
	listTargetDirs = [osp.join(strFlowDir, osp.basename(osp.normpath(sd))) for sd in listSourceDirs]
	intDirsTotal = len(listSourceDirs)

	moduleNetwork = Network(strModel).cuda().eval()

	intStartIndex = 0
	for intIndex, source_dir, target_dir in zip(list(range(1, len(listSourceDirs) + 1))[intStartIndex:],
												listSourceDirs[intStartIndex:], listTargetDirs[intStartIndex:]):
		print('[{}/{}]Processing: {}'.format(intIndex, intDirsTotal, osp.basename(osp.normpath(source_dir))))
		process_dir(moduleNetwork, strInputDir=source_dir, strOutputDir=target_dir, strFramesExt=strFramesExt)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Calculates optical flow through PWC-net')
	parser.add_argument('--model', dest='strModel', type=str, default='default', help="Model's name")
	parser.add_argument('--framesDir', dest='strFramesDir', type=str, help='Path to input dir with frames')
	parser.add_argument('--framesExt', dest='strFramesExt', type=str, default='.jpg', help='Frames file extension')
	parser.add_argument('--flowDir', dest='strFlowDir', type=str, help='Dir path where flow will be saved')
	args = parser.parse_args()

	process_frames_dir(args.strModel, args.strFramesDir, args.strFlowDir, args.strFramesExt)
