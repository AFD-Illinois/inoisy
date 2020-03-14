# input should be name of folder, without the '/' at end
# followed by raw or env

if [ "$1" != "" ] && [ "$2" != "" ]; then
		ffmpeg -r 24 -f image2 -i $1/%d_$2.png -vcodec libx264 -pix_fmt yuv420p -crf 25 $(dirname $1)/$(basename $1)_$2.mp4
else
		echo "Folder path or data type is empty"
fi
