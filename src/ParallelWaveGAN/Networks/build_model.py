#coding:utf-8
from .parallel_wavegan import ParallelWaveGANGenerator, ParallelWaveGANDiscriminator
from .melgan import MelGANGenerator, MelGANMultiScaleDiscriminator

def build_model(model_params={'generator':{}, 'discriminator':{}},
                generator_type="ParallelWaveGANGenerator", discriminator_type="MelGANMultiScaleDiscriminator"):

    generator = globals()[generator_type](**model_params['generator'])
    discriminator = globals()[discriminator_type](**model_params['discriminator'])

    return generator, discriminator
