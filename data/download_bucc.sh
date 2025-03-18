base_dir=./bucc2018/

# 如果已经下载过了，就不再下载
if [ -f $base_dir/.completed ]; then
    echo "Data was already downloaded at $DIR/bucc2018"
    exit 0
fi

mkdir $base_dir
for lg in zh ru de fr; do
    wget https://comparable.limsi.fr/bucc2018/bucc2018-${lg}-en.training-gold.tar.bz2 -q --show-progress
    tar -xjf bucc2018-${lg}-en.training-gold.tar.bz2
    wget https://comparable.limsi.fr/bucc2018/bucc2018-${lg}-en.sample-gold.tar.bz2 -q --show-progress
    tar -xjf bucc2018-${lg}-en.sample-gold.tar.bz2
done
mv $base_dir/*/* $base_dir/
for f in $base_dir/*training*; do mv $f ${f/training/test}; done
for f in $base_dir/*sample*; do mv $f ${f/sample/dev}; done
rm -rf $base_dir/*test.gold $base_dir/{zh,ru,de,fr}-en/
rm -rf bucc2018*tar.bz2

# 创建完成tag
echo "bucc2018" > $base_dir/.completed

echo "Successfully downloaded data at $DIR/bucc2018" 