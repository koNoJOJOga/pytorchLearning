想用 gitignore 忽略文件，必须先把它们从staged中移除：commit你已有的改变，保存当前的工作。

```bash
git rm --cached file/path/to/be/ignored。
git add .
git commit -m "fixed untracked files"
```

当文件夹是目录时，需要使用 -r 参数（递归）否则会报错

```bash
git rm -r --cached .
git add .
git commit -m "update .gitignore"  // windows使用的命令时，需要使用双引号
```
