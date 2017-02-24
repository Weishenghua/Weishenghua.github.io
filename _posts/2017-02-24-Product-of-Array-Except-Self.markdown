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
这个问题最直观的选项就是计算出所有数连乘的结果然后除以其中每一个数，但是存在的问题就是有可能会包含0；另外
题目也要求不能使用除法。</br>
时间复杂度和空间复杂度都为O(N)的解法是创建两个数组，FromBegin数组中的第i个元素表示从开始到i的所有元素的乘积，FromEnd这个数组中的第j个元素表示从结束开始到i的所有元素的乘积，计算出这两个数组的值，然后利用这两个数组的值来计算出结果。在计算FromBegin和FromEnd可以每次叠加的乘，所以时间复杂度为O（N）.</br>
时间复杂度为O（N）而空间复杂度为O（1）的方法是遍历整个数组nums，设结果数组为result,其中每一个元素初始化为 1 。在遍历数组的过程中，当索引为i时，from_begin = from_begin * nums[i]，from_end = from_end * nums[N-i-1]，这里的from_begin = FromBegin[i] = nums[0] * nums[1] * ... * nums[i-1], from_end = FromEnd[N-i-1] = nums[N-i-1] * nums[N-i-2] * ... * nums[i+1], 由于 result[i] = FromBegin[i] * FromEnd[i], 所以 result[i] * = from_begin，res[N-i-1]*=from_end。这样整个数组遍历完以后计算也就完成了。
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
