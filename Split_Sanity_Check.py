import os
import shutil
import random


# Source directory
source_men_dir = '/home/jerlshin/Documents/My_Work/__DATASET__/Men and Women Classification/men'
source_women_dir = '/home/jerlshin/Documents/My_Work/__DATASET__/Men and Women Classification/women'


# Destination directory for sanity check
destination_dir = '/home/jerlshin/Documents/My_Work/Generative Adversarial Networks Specialization/C3 Apply GAN - 26 hrs/Sanity_Check'

os.makedirs(os.path.join(destination_dir, 'men'), exist_ok=True)
os.makedirs(os.path.join(destination_dir, 'women'), exist_ok=True)

num_samples = 50

men_samples = random.sample(os.listdir(source_men_dir), num_samples)
women_samples = random.sample(os.listdir(source_women_dir), num_samples)

for men_sample in men_samples:
    shutil.copy(os.path.join(source_men_dir, men_sample), os.path.join(destination_dir, 'men', men_sample))

for women_sample in women_samples:
    shutil.copy(os.path.join(source_women_dir, women_sample), os.path.join(destination_dir, 'women', women_sample))

print("Done with split Freak!!")