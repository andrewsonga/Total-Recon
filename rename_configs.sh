config_dir=configs
current_config_names=("catpika-dualrig-fgbg001-depwt5-densetrunceikonalwt0p001-leftcam-finetune-fgbgkd" \
                      "catpika-dualrig-fgbg-finalvid000-leftcam-finetune-fgbgkd" \
                      "catamelie-dualrig003-freezebkgdshape-densetrunceikonalwt0p001-depwt5-entwt0-lr0p0001-leftcam-finetune-fgbgkd" \
                      "catamelie-dualrig-full000-freezebkgdshape-densetrunceikonalwt0p001-depwt5-entwt0-lr0p0001-leftcam-finetune-fgbgkd" \
                      "catsama-dualrig-fgbg000-depwt5-densetrunceikonalwt0p001-leftcam-finetune-fgbgkd" \
                      "dog-dualrig001-densetrunceikonalwt0p001-freezebkgdshape-eikonallossinert-lr0p0001-leftcam-finetune-fgbkgd" \
                      "dog-dualrig-fgbg000-depwt5-densetrunceikonalwt0p001-leftcam-finetune-fgbgkd" \
                      "human-dualrig002-leftcam-freezebkgdshape-densetrunceikonalwt0p001-depwt1p5-entwt0-equalsample-lr0p0001-finetune-fgbkgd" \
                      "humanhouse-dualrig-fgbg000-depwt5-densetrunceikonalwt0p001-leftcam-finetune-fgbgkd" \
                      "humandog-dualrig-thirdtry000-freezebkgdshape-eikonallossinert-lr0p0001-leftcam-finetune-fgbkgd" \
                      "humancat-dualrig-leftcam-freezebkgdshape-equalsample-lr0p0001-finetune-fgbkgd")
desired_config_names=("cat1-stereo000-leftcam-jointft" \
                      "cat1-stereo001-leftcam-jointft" \
                      "cat2-stereo000-leftcam-jointft" \
                      "cat2-stereo001-leftcam-jointft" \
                      "cat3-stereo000-leftcam-jointft" \
                      "dog1-stereo000-leftcam-jointft" \
                      "dog1-stereo001-leftcam-jointft" \
                      "human1-stereo000-leftcam-jointft" \
                      "human2-stereo000-leftcam-jointft" \
                      "human1dog1-stereo000-leftcam-jointft" \
                      "human2cat1-stereo000-leftcam-jointft")

for i in "${!current_config_names[@]}"; do
    current_config_name_leftcam=${current_config_names[$i]}
    desired_config_name_leftcam=${desired_config_names[$i]}
    current_config_name_rightcam=$(echo $current_config_name_leftcam | sed 's/leftcam/rightcam/g')
    desired_config_name_rightcam=$(echo $desired_config_name_leftcam | sed 's/leftcam/rightcam/g')

    cp $config_dir/$current_config_name_leftcam.config $config_dir/$desired_config_name_leftcam.config
    cp $config_dir/$current_config_name_rightcam.config $config_dir/$desired_config_name_rightcam.config

    #echo $current_config_name_leftcam
    #echo $current_config_name_rightcam
    #echo $desired_config_name_leftcam
    #echo $desired_config_name_rightcam
done