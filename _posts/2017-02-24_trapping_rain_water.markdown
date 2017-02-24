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
## 分析：

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
