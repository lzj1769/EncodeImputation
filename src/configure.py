training_data_csv = "/hpcwork/izkf/projects/ENCODEImputation/data/training_data/metadata_training_data.tsv"
training_data_loc = "/hpcwork/izkf/projects/ENCODEImputation/input/training_data"

validation_data_csv = "/hpcwork/izkf/projects/ENCODEImputation/data/validation_data/metadata_validation_data.tsv"
validation_data_loc = "/hpcwork/izkf/projects/ENCODEImputation/input/validation_data"

model_loc = "/hpcwork/izkf/projects/ENCODEImputation/model"

# We restrict this challenge to chromosomes 1-22 and
# chromosome X (i.e. ignore any data on chrY and chrM)
chrom_list = ['chr1', 'chr2', 'chr3',
              'chr4', 'chr5', 'chr6',
              'chr7', 'chr8', 'chr9',
              'chr10', 'chr11', 'chr12',
              'chr13', 'chr14', 'chr15',
              'chr16', 'chr17', 'chr18',
              'chr19', 'chr20', 'chr21',
              'chr22', 'chrX']

chrom_size_dict = {'chr1': 248956422,
                   'chr2': 242193529,
                   'chr3': 198295559,
                   'chr4': 190214555,
                   'chr5': 181538259,
                   'chr6': 170805979,
                   'chr7': 159345973,
                   'chr8': 145138636,
                   'chr9': 138394717,
                   'chr10': 133797422,
                   'chr11': 135086622,
                   'chr12': 133275309,
                   'chr13': 114364328,
                   'chr14': 107043718,
                   'chr15': 101991189,
                   'chr16': 90338345,
                   'chr17': 83257441,
                   'chr18': 80373285,
                   'chr19': 58617616,
                   'chr20': 64444167,
                   'chr21': 46709983,
                   'chr22': 50818468,
                   'chrX': 156040895}
