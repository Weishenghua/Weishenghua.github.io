---
layout: post
title:  "Leetcode: Trapping Rain Water"
date:   2017-02-24
categories: leetcode cpp 
tags: leetcode cpp
---

### Leetcode: Trapping Rain Water
## 题目要求：
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
For example, 
Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.
![image01](/assets/img/rainwatertrap.png)
图示中蓝色部分表示累积的雨水。
## 分析：
这个问题和之前的Container with most water（<https://weishenghua.github.io/leetcode/cpp/2017/02/24/leetcode-Container_With_Most_Water.html>） 比较类似，但是要求的是整个等高线图的全部储水量。这个问题这个问题仍然可以在O（N）的时间复杂度中解决。首先初始化总储水量为0，设置左右两个游标，两个游标形成一个容器，这个容器的宽度为 right - left, 高度为min(height[right],height[left]), 可以算出在当前这个高度和宽度下的储水量。然后往中间寻找左右游标，寻找一个高度比之前容器高的容器。计算出这个容器储水量，并且累积在之前的总储水量上，在这里累积时需要注意，只能累积超出上一个容器高度那部分的容量，否则会重复累积。整个累积过程结束后，由于在这个问题里bar也是占据空间的，所以我们需要用累积的总储水量减掉所有bar占的体积之和，就得到了实际的总储水量。
## Code:
{% highlight c++ lineno %}
class Solution {
public:
    int trap(vector<int>& height) {
        int left = 0;
        int right = height.size()-1;
        int water = 0;
        int temp_h = 0;
        while(left<=right)
        {
            int h = min(height[left],height[right])-temp_h;
            temp_h = min(height[left],height[right]);
            int c = right - left + 1;
            water += h*c;
            while(height[left]<=temp_h&&left<=right)
            {
                left++;
            }
            while(height[right]<=temp_h&&left<=right)
            {
                right--;
            }
            
        }
        int block = 0;
        for(int i =0;i<height.size();i++)
        {
            block= block + height[i];
        }
        water = water - block;
        return water;
        
    }
};
{% endhighlight %}
