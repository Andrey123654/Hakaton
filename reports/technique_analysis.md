# Анализ техник фишинга и объяснение модели

Обучающая выборка: **63900** URL (фишинг: 23982, безопасные: 39918).

## 1. Распространённость техник атак

`lift` = доля признака среди фишинга / доля среди безопасных URL. Значение > 1 означает, что признак чаще встречается у фишинга.

| Техника | Доля у фишинга | Доля у безопасных | lift | Найдено у фишинга |
|---|---|---|---|---|
| Punycode (xn--) | 0.17% | 0.00% | 1668.9 | 40 |
| IP-адрес вместо домена | 1.42% | 0.01% | 186.7 | 341 |
| Много поддоменов (>=3) | 6.88% | 0.20% | 35.2 | 1650 |
| «Плохая» зона (.tk/.xyz/.top...) | 10.30% | 0.34% | 30.7 | 2469 |
| Бесплатный хостинг | 7.42% | 0.81% | 9.2 | 1779 |
| Слова verify/confirm/update | 4.95% | 0.74% | 6.7 | 1188 |
| Бренд в поддомене | 8.91% | 1.69% | 5.3 | 2137 |
| Двойное расширение (.jpg.php) | 0.03% | 0.01% | 3.8 | 7 |
| Слова login/signin | 16.17% | 4.66% | 3.5 | 3878 |
| Ключевые слова-приманки (>=2) | 15.04% | 4.66% | 3.2 | 3607 |
| Доставка (parcel/tracking/order) | 0.81% | 0.45% | 1.8 | 195 |
| Финансовые слова (pay/bank/wallet) | 11.35% | 6.51% | 1.7 | 2721 |
| Мимикрия: бренд вне домена | 31.77% | 19.16% | 1.7 | 7620 |
| Бренд с добавкой/дефисом (sberbank-online) | 0.38% | 0.29% | 1.3 | 92 |
| Длинный URL (>75 символов) | 26.72% | 21.46% | 1.2 | 6409 |
| Сокращатель ссылок | 7.37% | 7.03% | 1.0 | 1767 |
| Конструкция user@host | 0.00% | 0.00% | 1.0 | 0 |
| Гомоглифы (paypa1, rnicrosoft) | 0.00% | 0.00% | 1.0 | 0 |
| Бренд в пути | 15.33% | 17.26% | 0.9 | 3677 |
| Процентное кодирование | 2.26% | 2.79% | 0.8 | 542 |
| Призы и бонусы (bonus/prize/free) | 1.81% | 2.51% | 0.7 | 433 |
| Тайпсквоттинг (опечатка в домене) | 1.04% | 2.36% | 0.4 | 250 |
| Нестандартный порт | 0.15% | 0.43% | 0.4 | 37 |

## 2. Тайпсквоттинг: домены-двойники

| url | imitates | distance |
|---|---|---|
| http://www.bancm.co/desbloqueos/USERscis=login-Dno-back-button.html | ('bank', 2) | 2 |
| https://www.paypalil.ga/ | ('paypal', 2) | 2 |
| http://www.grupwa-mabaryoutuber.otzo.com/login.php | ('ozon', 2) | 2 |
| http://www.whatsappjoingrup.zyns.com/login.php | ('dns', 2) | 2 |
| https://hood.ae/assets/img/mkbbssl/e9750b47bedf994ddc9a96889649ea36/ | ('zoom', 2) | 2 |
| http://kadubeureum-serang.desa.id/wp-content/auto/autolink/autolinkauto/mailboxx/mailbox/index.php?email=jifra | ('visa', 2) | 2 |
| https://zorg.cl/auth/?platform=hootsuite | ('zoom', 2) | 2 |
| https://doccomo.ne.uqay.xyz | ('ebay', 2) | 2 |
| https://www.cr.mufg.acoe.cc/ | ('adobe', 2) | 2 |
| http://yamabito.main.jp/ | ('mail', 1) | 1 |
| https://doccomo.ne.eqae.xyz | ('ebay', 2) | 2 |
| http://secureb101-verify.ddns.us/true/ | ('dns', 1) | 1 |

## 3. Гомоглифные подмены

_Примеры не найдены._

## 4. Мимикрия под бренд (бренд вне домена)

| url | imitates |
|---|---|
| https://www.grupwhatsapnadia22.duckdns.org/ |  |
| https://sbccgloball.net/dropbox/Rich/main/en.php?common_l&wp=W_est&sub&e=36297d997113646d9629b73a0b11c9b69d2a7 | dropbox |
| https://supperbelle.com/Raiffeisen.html | raiffeisen |
| https://clinicavillavitoria.com.br/@@/fb/ | avito |
| https://basis.org.ua/sites/all/themes/omega/alpha/chartererror/error.php |  |
| http://lloydsbank.online-securedata-personal.com/Login.php | bank |
| https://helpsresources.com/book/cibc/confirm.php |  |
| http://verificasihfacebooks.ga/ | facebook |
| http://563908.selcdn.ru/round/boxoffice1.html | office |
| http://bityska.test-unifer.eu/wp-content/next.php?ss=2&amp;email=bmV3LmNsYWltc25vdGljZXNAamFtZXNyaXZlcmlucy5jb | mail |
| https://www.authorise-myonlinebankingdetails.com/lloyds/Login.php | bank |
| https://mail.dhlesgmarketing.com/2x | mail |

## 5. Ключевые слова-приманки

| url | n_phish_words |
|---|---|
| https://www.securepayment.icu/Login.php | 3 |
| http://octango.net/cg/suntrust.login.ssl/e6deb196faa2d14bb076ac880ebcec2f/confirm.php?cmd=login_submit&amp;id= | 4 |
| http://securemyacc.com/login.php?cmd=login_submit&amp;id=7d7d78a6c46222f8027216092b8b2b207d7d78a6c46222f802721 | 4 |
| https://bootnbonnet.org/wp/wp-includes/Text/excel/excel/excel/login.php?cmd=login_submit&amp;id=cd898a1e0c7968 | 3 |
| https://payeesecurity-support.com/halifax/ | 3 |
| https://www.baby2sleep.co.uk/home/chas3-us-en-login/signin.php?cmd=_chase-signin=&dd9c54662546a78ed541e6c5f70b | 4 |
| http://thewintershops.com/wp-admin/-/account_americafirstCU_login/login.php?cmd=login_submit&amp;id=MTY0NjUxOD | 5 |
| http://rulomatik.hu/images/confirmnewboa/confirmnewboa/login.php?cmd=login_submit&amp;id=0ca838e9ad29661397c32 | 4 |
| http://www.elkinsconstruction.com/rossy/dropboxverification/a9c0ba3641f348ee7b2e48260101609b/login.php?cmd=log | 4 |
| http://vieuxshack.com/download/adobe/de28191eb90b96e9ba93d889fc37511b/login.php?cmd=login_submit&amp;id=a1ca86 | 3 |
| http://zohard.com/.well-known/pki-validation/mod/unblockverification/customer_center/customer-IDPP00C963/signi | 6 |
| https://ssmbcss.vv4atpvm.cc/pc/index.php?openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.smbc.co | 3 |

## 6. IP-адрес вместо домена

| url |
|---|
| http://51.13.103.32/reconfirm.php?sf58gfd1s689sxd2sdf8angf264s9df23sd2f1n495K3L2C151645172991f1477dbd26917ef38 |
| http://200.107.61.17/CIBI/cibc/ |
| http://69.167.151.209/files/e3e58bac6d641e604fe7e289537b2b16/ |
| http://47.90.0.38/login/home |
| http://191.232.190.147/produto199426true/produto.php?linkcompleto=https://www.americanas.com.br/produto/165449 |
| http://221.150.115.216/wordpress/wp-content/php/81b44841fd564c347f7f21ae19b97659/ |
| http://69.167.151.209/files/bb6b869c6775b196843c7f534fbc0a63/ |
| http://221.150.115.216/wordpress/wp-content/php/096b826c74d0c38d76aa6dbae35ed59d/ |
| http://69.167.151.209/files/ffa74abd380bd12f9829f639663697f5/details.php |
| http://167.99.52.152/mobile/login.php?enc=1 |
| http://190.211.254.196/banks/Simplii/ebm-mobile-app/index.html |
| http://190.211.254.196/banks/Scotia/?securessl=true&amp;sessionid=5e639134cbdf89db4243214a433b2426 |

## 7. Ошибки модели на OOF-предсказаниях


## 7.1. Ложноположительные (безопасный сайт признан фишингом)

| url | proba |
|---|---|
| http://sagenerators.co.za/verkaufer5f72c45f31da65eb3cb1576cb136c12344aa49ae4272417cfdeinvoicerouteacctpagetype | 0.9999992488089035 |
| http://4c8addca221efe1d7977ec9ca3ddb148.registreimarcasepatentes.com.br/ | 0.9999751983142056 |
| http://18yr-old-teen-pounde26.conditi73indulge.dance/ | 0.9999445024654825 |
| http://ugly-brunette-abcy-d88.isle27alverta.webcam/ | 0.9998770374893378 |
| http://playamayayucatan.com/wp-includes/Requests/Auth/ | 0.9997395719357893 |
| https://adobeacrobat.app.link/FYb9qvrmG0 | 0.9993097286508615 |
| http://crqsc.aftonline.com.br/ | 0.998785656468561 |
| https://ubipayroll.com/ESKA/index.php/apemc/login | 0.997376455085245 |
| https://d34vs84kik5724.cloudfront.net/ | 0.9972052083361819 |
| https://accounts.googleaccount.us.com/ServiceLogin?passive=1209600&amp;continue=https://docs.google.com/&amp;f | 0.9968725604696149 |
| http://webworkmoms.com/crwqmsx@secure.indeed.com/account/crwqmsx@hl=en_us&co=us&service=my&continue=https_253a | 0.9967831718220039 |
| https://adamscott21201.wixsite.com/ethernetclocks | 0.9963703560249095 |
| http://www.security-explorations.com/ | 0.995819793579528 |
| https://fish.ebanksepah.ir/ | 0.993743277603807 |
| http://getadating.com/ | 0.992616346874976 |

## 7.2. Ложноотрицательные (фишинг пропущен)

| url | proba |
|---|---|
| https://www.fischerpanda.co.uk/Fischer_Panda_UK_Generators_Secure_Lloyds_Register_Type_Approval.html | 7.71846115641079e-06 |
| https://www.namikeystonepa.org/xanax/ | 3.7627429783945714e-05 |
| https://realvcc.com/product-tag/paypal-vba-micro-deposit-us-virtual-bank/ | 0.00011969137437029712 |
| https://kergaukr.com/feed/posts/chelsea-clinton-calls-out-national-enquirer-for-attacks-against-her-mother | 0.00012064555262452913 |
| https://cyberrune.com/forum/ | 0.00014058934844919077 |
| http://www.tudosobretudo.blog.br/banco-bradesco-agencia-endereco/bradesco-agencia-6621-shopping-granja-viana-r | 0.00019530542286504623 |
| https://www.dhlonboarding.com/auth.cfm?sign_in=1&forcemobile=false | 0.00021310677741472595 |
| https://applebandsmart.com/ | 0.0002261983950160074 |
| https://powercase.shoplineapp.com/products/%E2%98%98%EF%B8%8F%E3%80%90%E4%B8%89%E8%91%89%E8%8D%89%E3%80%91%E3% | 0.0002341204674312738 |
| http://www.hide-windows.com/free_youtube_downloader/youtube_converter.htm | 0.0004581861119068982 |
| https://mcaremart.com/buy-hydrocodone-online.html | 0.0005312585403944467 |
| https://linklyhq.com/account-over-limit | 0.0007855755880359283 |
| https://www.kosmetica.com.bd/skincare-store/img/header/myzombi/home/signin/ | 0.0008919547763520033 |
| https://icocliqs.com/cash-app-carding/ | 0.001417849734698094 |
| https://shewanted.com/ | 0.001487562324620095 |

## 8. Графики

![distributions.png](figures/distributions.png)

![techniques.png](figures/techniques.png)

![importances.png](figures/importances.png)

![oof_errors.png](figures/oof_errors.png)
