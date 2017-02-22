---
layout: post
title:  "Leetcode: 3 sum"
date:   2017-02-22
categories: leetcode cpp 
tags: leetcode cpp
---

### Leetcode: 3 sum
## 题目要求：
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note: The solution set must not contain duplicate triplets.
## Example:
For example, given array S = [-1, 0, 1, 2, -1, -4],
A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
## 分析：
题目要求在一个数组中找出a+b+c=0的三个数，并且要求a,b,c组成不重复的三元组。类似于2 sum问题，首先对于数组整体进行排序，然后遍历整个数组，在每次迭代时固定num1，令 target = 0-num1,然后在其他的数字中寻找num2+num3=target的组合，这部分可以参照2 sum问题。其中比较困难的一点就是要保证三元组的不重复性，解决的方案就是不论对于num1,num2或者num3,保证在迭代中下一个取值要和上一个取值不一样即可。算法的时间复杂度为O(N^2).
## Code:
{% highlight c++ lineno %}
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        if (nums.size()==0)
        {
            return res;
        }
        sort(nums.begin(),nums.end());
        for (int i =0; i<nums.size(); i++)
        {
            int left = i+1;
            int right = nums.size()-1;
            int target = 0 - nums[i];

               while(left<right)
                {
                    if(nums[left]+nums[right]<target)
                    {
                        left++;
                    }
                    else
                    {
                        if(nums[left]+nums[right]>target)
                        {
                            right--;
                        }
                        else
                        {

                            vector<int> input;
                            input.push_back(nums[i]);
                            input.push_back(nums[left]);
                            input.push_back(nums[right]);
                            res.push_back(input);
                            //remove duplicate left
                            while(left<right&&nums[left]==input[1]) 
                            {
                                left++;
                            }
                            //remove duplicate right
                            while(left<right&&nums[left]==input[2]) 
                            {
                                right--;
                            }
                        }
                                        
                    }
                    
                }
          //remove duplicate i
           while (i + 1 < nums.size() && nums[i + 1] == nums[i]) 
            {
                i++;
            }
        
        }
        return res;
    }
};
{% endhighlight %}

