src_dir=/data/ndsong/data/banmo_ds

for files in `ls -d $src_dir/*`; do
    if [ $(basename $files) != "database" ] && [ $(basename $files) != "logdir" ]; then
        echo $(basename $files)
        cp -r $files ./
    fi
done

sudo cp -r $src_dir/.git ./
sudo cp -r $src_dir/.gitignore ./
sudo cp -r $src_dir/.gitmodules ./