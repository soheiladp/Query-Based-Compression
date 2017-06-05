#/bin/sh

echo 'Complie the C++ utility library'
python setup.py build_ext -b ./cutils/ --swig-cpp

echo 'Setup Thirdparty packages'
cd ./thirdparty

echo ' -- Acs12'
cd Acs12
./setup.sh
cd ..

echo ' -- xiaokui'
cd xiaokui
./setup.sh
cd ..

cd ..
