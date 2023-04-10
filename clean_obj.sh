for testdir in `ls -d logdir/*`; do

    # list of experiment directories whose .obj files we DON'T want to erase
    if [ $testdir != "logdir/catamelie-dualrig002-properlyscaledcams-leftcam-ds-finetune-fgbkgd" ] && [ $testdir != "logdir/catamelie-dualrig002-properlyscaledcams-leftcam-ds-silwt0-finetune-fgbkgd" ] && [ $testdir != "logdir/human-dualrig002-properlyscaledcams-leftcam-ds-finetune-fgbkgd" ] && [ $testdir != "logdir/human-dualrig002-properlyscaledcams-leftcam-ds-silwt0-finetune-fgbkgd" ]
    then 
        # testdir has format "logdir/..."
        echo deleting obj files, checkpoints, rendered images inside $testdir

        # deleting mesh*.obj and bone*.obj generated during generating visualizations with render_nvs.sh
        find $testdir -not -name 'mesh*' -not -name 'bone*' -name '*.obj' -delete

        # deleting the intermediate checkpoints generated during training
        find $testdir -not -name 'params_latest*' -name '*.pth' -delete

        # deleting the images generated during evaluation
        find $testdir -name 'nvs-*' -name '*.png' -delete
    fi
done
