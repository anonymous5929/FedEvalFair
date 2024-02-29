#!/bin/bash
for rate in {1..9} ; do
	i=0
	list=()
	j=0
	for path in `find eicu_DP_0-01_0-50_multi_0-${rate}_*/results/new_trial/ | sort` ; do
		if [ ${i} -eq 200 ] ; then
			if [ -d ${path} ] ; then continue; fi
                        echo ${path}
			echo ${list[$j]}.xlsx
                        mkdir -p train_rate2/eicu_DP_0-01_0-50_multi_0-${rate}/
                        cp ${path} train_rate2/eicu_DP_0-01_0-50_multi_0-${rate}/${list[$j]}.xlsx
			j=$((j+1))
			if [ ${j} -eq ${#list[@]} ] ; then break; fi
		else
			if [ ! \( -e train_rate/eicu_DP_0-01_0-50_multi_0-${rate}/${i}.xlsx \) ] ; then
				list+=("${i}")
                	fi
			i=$((i+1))
		fi
	done
done
