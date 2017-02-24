---
layout: post
title:  "Leetcode: Container With Most Water"
date:   2017-02-24
categories: leetcode cpp 
tags: leetcode cpp
---

### Leetcode: Container With Most Water
## 题目要求：
Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.
## 分析：
这道题目的要求寻找最大的容器的容积是多少。首先设置左右两个游标，从最左边和最右边分别开始向中间滑动，当前的容器容积由容器的宽 right-left 和 容器的高度
min(height[left],height[right])决定，计算出当前容器的容积；在后续游标滑动的过程中，由于两个游标都往中间滑动，所以容器的宽必然会减少，所以在分别滑动左右游标的同时，游标要一直滑动到比当前的
min(height[left],height[right])还要大的时候再停止，重新计算容器的容积，这样的容器容积才有可能会比之前最大的容器容积大
## Code:
{% highlight c++ lineno %}
class Solution {
public:
    int maxArea(vector<int>& height) {
        int left = 0;
        int right = height.size()-1;
        int res = (right-left)*min(height[left],height[right]);
        while(left<right)
        {
            int width = right-left;
            int c_height = min(height[left],height[right]);
            int area = width*c_height;
            if(area>res)
            {
                res = area;
            }
            while(left<right&&height[left]<=c_height)
            {
                left++;
                
            }
            while(right>left&&height[right]<=c_height)
            {
                right--;
                
            }
        }
        return res;
    }
};
{% endhighlight %}
