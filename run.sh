python pg.py --test --use-cuda --test-img-dir "{}" --test-prior-dir "{}" > test.log # run inference to select priors
python parse.py --img_dir "{}" --prior_dir "{}" --logfile test.log --name filter_gym_ir # collect the selected riors to a new folder "filter_gym_ir"
python main.py --use-cuda --ca # domian adaptation with priors as ground truth, should change the data path in file 
