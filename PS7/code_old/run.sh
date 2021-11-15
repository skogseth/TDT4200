generate_video () {
    a=1
    for i in $( ls -1v ${OUTPUT_PATH}output-* ); do
        new=$(printf "${OUTPUT_PATH}%03d.png" "$a")
        mv -i -- "$i" "$new"
        let a=a+1
    done
    ffmpeg -framerate 30 -i ${OUTPUT_PATH}%03d.png output.mp4
}

IMG1="man9"
IMG2="man10"
#IMG1="woman5.jpg"
#IMG2="woman8.jpg"
ROOTDIR="."
IMG_PATH="./images"
LINE_PATH="./lines/lines-${IMG1}-${IMG2}"
OUTPUT_PATH="./output-png/"
STEPS=90

make
rm ${OUTPUT_PATH}*.png
cmd="${ROOTDIR}/morph ${IMG_PATH}/${IMG1}.jpg ${IMG_PATH}/${IMG2}.jpg ${LINE_PATH}.txt ${OUTPUT_PATH} ${STEPS}"
echo $cmd
eval "$cmd"

generate_video
