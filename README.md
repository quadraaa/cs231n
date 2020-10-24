# cs231n
Useful commands:

docker run -v $(pwd):/src/notebooks -p 8888:8888 -td okwrtdsh/anaconda3:pytorch-cpu

vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages\s+([^:]+)[^\d]+(\d+)/ and printf("%-16s % 16.2f Mi\n", "$1:", $2 * $size / 1048576);'