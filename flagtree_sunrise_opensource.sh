#!/bin/bash

# 这个脚本用来把 gitlab 的flagtree导出为开源版本的flagtree
# 【注意】这个脚本会把目标目录内的.git以外的内容都删除

function print_usage() {
    echo "flagtree_sunrise_opensource.sh <github_flagtree_proj_dir>"
}

if [ $# -ne 1 ] ; then
    print_usage
    exit 1
fi

git remote -v | grep 'https://gitlab.sunrise-ai.com/stpu-rt/flagtree.git' > /dev/null
if [ $? -ne 0 ] ; then
    echo "this script must be executed in flagtree root diectory!"
    exit 1
fi

if [ ! -d $1 ] ; then
    echo "github_flagtree_proj_dir not exit!"
    exit 1
fi

workdir=`pwd`
cd $1
opensource_dir_check_ret=1
git remote -v | grep 'https://github.com/' | grep flagtree.git > /dev/null
opensource_dir_check_ret=$?
cd $workdir

if [ $opensource_dir_check_ret -ne 0 ] ; then
    echo "$1 is not a opensource github flagtree project"
    exit 1
fi

rm -rf $1/*
cp -rf ./* $1/
cp -rf .github $1/
cp -rf .clang-format $1/
cp -rf .editorconfig $1/
cp -rf .gitignore $1/
cp -rf .pre-commit-config.yaml $1/

rm -rf third_party/sunrise/plugin
