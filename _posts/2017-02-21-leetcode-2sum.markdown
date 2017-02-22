---
layout: post
title:  "Leetcode: 2 sum"
date:   2017-02-21
categories: leetcode cpp 
tags: leetcode cpp
---

### Leetcode: 2 sum
## 题目要求：
Given an array of integers, return indices of the two numbers such that they add up to a specific target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
## Example:
Given nums = [2, 7, 11, 15], target = 9,
Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
## 分析：
题目要求在一个数组中找出两个数，这两个数的和等于一个指定的数target.使用简单的思路可以达到o(n^2)的复杂度。如果将原来的数组进行排序，使用一对左右游标从数组的两侧从中间滑动，当当前两个数之和大于target时，右游标往左滑动，当两个数之和大于target时，左游标往右滑动，最终可以得到结果。并且可以证明左右游标的这种滑动方式是不会错过正确结果的。
## Code:
{% highlight c++ %}
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        // new o(n) method
        vector<int> res;
        if(nums.size()==0)
        {
            return res;
        }
        
        vector<int>  idx(nums.size());
        for (int i = 0; i < idx.size(); i++) 
        {
            idx[i] = i;
        }
         // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(),[&nums](int i1, int i2) {return nums[i1]<nums[i2];});
         // sort original nums array
        sort(nums.begin(),nums.end());
        
        int left = 0;
        int right = nums.size()-1;
        while(left!=right)
        {
            if(nums[left]+nums[right]>target)
            {
                right--;
            }
            else
            {   if(nums[left]+nums[right]<target)
                    {
                        left++;
                    }
                else
                {
                    res.push_back(idx[left]);
                    res.push_back(idx[right]);
                    return res;
                }
            }
            
        }
        
    }
};
{% endhighlight %}
## 另一种思路：
对于两个数字求和的问题，还可以将每一个数字存储到hashtable中，然后遍历整个数组，查找hashtable 中是否包含 target - current_num这个数字，每一次查询只需要O(1)的时间复杂度，总的时间复杂度为O(N).不过由于使用了hashtable，增加了额外的空间。
