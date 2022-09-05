array=( 500 1000 2000 3000 4000 5000)
#array=(3000 4000)
for i in "${array[@]}"
do
   ./main $i $i
done
