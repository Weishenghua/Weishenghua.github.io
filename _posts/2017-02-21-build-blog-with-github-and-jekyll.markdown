---
layout: post
title:  "简单利用github pages和jekyll配置个人主页"
date:   2017-02-21
categories: Jekyll
tags: Jekyll
---

Github 为广大用户提供了一个Github pages的功能，可以用于免费搭建个人的主页或者博客。 利用Github pages和 Jekyll相结合能够快速简单的对于个人的博客进行配置和管理，非常方便和实用，这里就我个人利用Github pages和 Jekyll来搭建博客的过程进行介绍。

 1. 在Github 上建立新的项目
 在自己github的主页上点击 "New Respository" ,进入项目初始化配置界面:
 ![image01](/assets/img/github_create_respositoty.png)
 其中在Repository name 这一栏填上 xxxx(你的用户名).guthub.io。
 Description 可以填一些博客的基本介绍信息。
 选择项目为公开"Public", 最好勾选"Initialize this repository with a README"。
 最后单击最下面的Create Respositoty.
 1. 利用Jekyll模板
 Jekyll是一个简单的免费的Blog生成工具,我们可以在网上找到一些现有的Jekyll模板并且在其基础上进行修改来实现自己的需求。
 首先将自己的项目克隆到本地,然后将网上下载的Jekyll模板解压到含有.git这个文件夹的文件夹下（.git可能会被隐藏显示）。一般来说博客文件夹的目录结构就会如下所示：
 ![image02](/assets/img/jekyll_file_directory.png)
 其中"post"文件夹中放一些自己要写的博客markdown文件，这些markdown文件应该命名为"YYYY-mm-dd-filename.markdown",如图所示:
 ![image03](/assets/img/_post_structure.png)
 另一个重要的文件是config.yml文件，其中包含了一些对于博客的配置信息，一些模板中会有title,email,description,github等选项，按照个人的需求进行填写。官方yml文件的配置说明请戳 <http://jekyllcn.com/docs/configuration/>
 1. 结果展示
 在浏览器中输入 https://yourusername.github.io/ 就可以浏览配置好的主页，这是我利用一个模板建立的主页(<https://github.com/P233/3-Jekyll>):
  ![image04](/assets/img/homepage.png)
