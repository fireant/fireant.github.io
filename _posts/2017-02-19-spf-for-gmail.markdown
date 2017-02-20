---
layout: post
title:  "The SPF value good for Google"
date:   2017-02-19 20:09:00 -0700
categories: Misc
---

After trying a few SPF values, the one that seems to convince Google the right server sending out has this template:


```
v=spf1 ip4:<ip_of_your_mail_server> include:<domain_of_your_mail_server> ~all 
```

