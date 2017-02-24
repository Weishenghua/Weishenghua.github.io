---
layout: post
title:  "Leetcode: Product of Array Except Self"
date:   2017-02-24
categories: leetcode cpp 
tags: leetcode cpp
---

### Leetcode: Product of Array Except Self
## 题目要求：
Given an array of n integers where n > 1, nums, return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

Solve it without division and in O(n).

For example, given [1,2,3,4], return [24,12,8,6].

Follow up:
Could you solve it with constant space complexity? (Note: The output array does not count as extra space for the purpose of space complexity analysis.)

Subscribe to see which companies asked this question.。
## 分析：
这个问题和之前的Container with most water（<https://weishenghua.github.io/leetcode/cpp/2017/02/24/leetcode-Container_With_Most_Water.html>） 比较类似，但是要求的是整个等高线图的全部储水量。这个问题这个问题仍然可以在O（N）的时间复杂度中解决。首先初始化总储水量为0，设置左右两个游标，两个游标形成一个容器，这个容器的宽度为 right - left, 高度为min(height[right],height[left]), 可以算出在当前这个高度和宽度下的储水量。然后往中间寻找左右游标，寻找一个高度比之前容器高的容器。计算出这个容器储水量，并且累积在之前的总储水量上，在这里累积时需要注意，只能累积超出上一个容器高度那部分的容量，否则会重复累积。整个累积过程结束后，由于在这个问题里bar也是占据空间的，所以我们需要用累积的总储水量减掉所有bar占的体积之和，就得到了实际的总储水量。
## Code:
{% highlight c++ lineno %}
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> res(nums.size(),1);
        int from_begin = 1;
        int from_end = 1;
        for(int i = 0; i<nums.size(); i++)
        {
            res[i]*=from_begin;
            res[nums.size()-i-1]*=from_end;
            from_begin = from_begin * nums[i];
            from_end = from_end * nums[nums.size()-i-1];

        }
        return res;
    }
};
{% endhighlight %}
