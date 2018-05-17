#!/bin/bash

n_replicas=281
wall_time=10:00
n_structures=1
variables=structures,norm
norm=6843
alpha=10
data_set=bothdomains
schedule=\\/scratch\\/scarste\\/ensemble_hic\\/bau2011\\/GM12878_new_smallercd_nosphere_40structures_sn_122replicas\\/analysis\\/te0.25_schedule.pickle
ensemble=boltzmann
gauss_mean=0.7
gauss_std=0.3
rate=0.05
suffix=
n_samples=50000
adaption_limit=100000000

data_file=${data_set}

if [ "$variables" == "structures,norm" ]; then
    opf_vars=sn
fi

if [ "$variables" == "structures" ]; then
    opf_vars=s
fi

if [ "$variables" == "structures,weights" ]; then
    opf_vars=sw
fi

output_folder=${data_set}${suffix}_${n_structures}structures_${n_replicas}replicas\\/

tstamp="$(date +%Y%m%d%H%M%S)"
tmpcfg=tmp/${tstamp}.cfg
tmpcfg_forsed=tmp\\/${tstamp}.cfg
tmpstart=tmp/${tstamp}.sh
cp myconfig.cfg $tmpcfg
cp mystart.sh $tmpstart

sed -i 's/n_structures_PH/'"$n_structures"'/g' $tmpcfg
sed -i 's/variables_PH/'"$variables"'/g' $tmpcfg
sed -i 's/data_file_PH/'"$data_file"'/g' $tmpcfg
sed -i 's/output_folder_PH/'"$output_folder"'/g' $tmpcfg
sed -i 's/norm_PH/'"$norm"'/g' $tmpcfg
sed -i 's/schedule_PH/'"$schedule"'/g' $tmpcfg
sed -i 's/n_samples_PH/'"$n_samples"'/g' $tmpcfg
sed -i 's/rate_PH/'"$rate"'/g' $tmpcfg
sed -i 's/n_replicas_PH/'"$n_replicas"'/g' $tmpcfg
sed -i 's/gauss_mean_PH/'"$gauss_mean"'/g' $tmpcfg
sed -i 's/gauss_std_PH/'"$gauss_std"'/g' $tmpcfg
sed -i 's/ensemble_PH/'"$ensemble"'/g' $tmpcfg
sed -i 's/norm_rate2_PH/'"$norm_rate"'/g' $tmpcfg
sed -i 's/norm_shape2_PH/'"$norm_shape"'/g' $tmpcfg
sed -i 's/n_beads_PH/'"$n_beads"'/g' $tmpcfg
sed -i 's/alpha_PH/'"$alpha"'/g' $tmpcfg
sed -i 's/adaption_limit_PH/'"$adaption_limit"'/g' $tmpcfg

sed -i 's/wall_time_PH/'"$wall_time"'/g' $tmpstart
sed -i 's/n_replicas_PH/'"$((n_replicas+1))"'/g' $tmpstart
sed -i 's/config_PH/'"$tmpcfg_forsed"'/g' $tmpstart
sed -i 's/sim_script_PH/'"$sim_script"'/g' $tmpstart

cp $tmpcfg tmpcfg.cfg

bsub < $tmpstart
