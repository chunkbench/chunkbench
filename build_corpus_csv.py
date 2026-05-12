"""
Creates corpus_list.csv for the ChunkBench GitHub repository.
Sources: Additional_file_1.docx (134 citations) + corpus_manifest.csv
Outputs: doc_id, corpus_role, citation_no, first_author, year, title, journal, doi, pubmed_link
"""
import csv, re, os

# ---------------------------------------------------------------------------
# 134 Vancouver citations from Additional_file_1.docx
# ---------------------------------------------------------------------------
CITATIONS_RAW = [
    (1,  "Bork K, Maurer M, Bas M, Hartmann K, Biedermann T, Kreuz W, et al. German Guideline for Hereditary Angioedema due to C1-INH Deficiency. Guideline of the German Society for Angioedema Research (DGA), et al. 2011."),
    (2,  "Lang DM, Aberer W, Bernstein JA, Chng HH, Grumach AS, Hide M, et al. International consensus on hereditary and acquired angioedema. Ann Allergy Asthma Immunol. 2012;109:395-402. doi: 10.1016/j.anai.2012.10.008"),
    (3,  "Björkqvist J, Sala-Cunill A, Renné T. Hereditary angioedema: a bradykinin-mediated swelling disorder. Thromb Haemost. 2013;109(3):368-74. doi: 10.1160/TH12-08-0549"),
    (4,  "Zuraw BL, Bernstein JA, Lang DM, Craig T, Dreyfus D, Hsieh F, et al. A focused parameter update: Hereditary angioedema, acquired C1 inhibitor deficiency, and angiotensin-converting enzyme inhibitor–associated angioedema. J Allergy Clin Immunol. 2013;131:1491-3. doi: 10.1016/j.jaci.2013.03.034"),
    (5,  "Longhurst HJ, Tarzi MD, Ashworth F, Bethune C, Cale C, Dempster J, et al. C1 inhibitor deficiency: 2014 United Kingdom consensus document. Clin Exp Immunol. 2015;180(3):475-83. doi: 10.1111/cei.12584"),
    (6,  "Andrejević S, Korošec P, Šilar M, Košnik M, Mijanović R, Bonači-Nikolić B, et al. Hereditary Angioedema Due to C1 Inhibitor Deficiency in Serbia: Two Novel Mutations and Evidence of Genotype-Phenotype Association. PLoS ONE. 2015;10(11):e0142174. doi: 10.1371/journal.pone.0142174"),
    (7,  "Bernstein JA, Riedl M, Zacek L, Shapiro RS. Facilitating home-based treatment of hereditary angioedema. Allergy Asthma Proc. 2015 Mar-Apr;36(2):92-9. doi: 10.2500/aap.2015.36.3820"),
    (8,  "Björkqvist J, de Maat S, Lewandrowski U, Di Gennaro A, Oschatz C, Schönig K, et al. Defective glycosylation of coagulation factor XII underlies hereditary angioedema type III. J Clin Invest. 2015;125(8):3132-46. doi: 10.1172/JCI77139"),
    (9,  "Bork K, Wulff K, Witzke G, Hardt J. Hereditary angioedema with normal C1-INH with versus without specific F12 gene mutations. Allergy. 2015;70(8):1004-12. doi: 10.1111/all.12648"),
    (10, "Joseph K, Bains K, Tholanikunnel BG, Bygum A, Aygören-Pürsün E, Bork K, et al. A novel assay to diagnose hereditary angioedema utilizing inhibition of bradykinin-forming enzymes. Allergy. 2015;70(1):115-9. doi: 10.1111/all.12521"),
    (11, "Moldovan D, Bernstein JA, Cicardi M. Recombinant replacement therapy for hereditary angioedema due to C1 inhibitor deficiency. Immunotherapy. 2015;7(8):887-900. doi: 10.2217/IMT.15.44"),
    (12, "Patel N, Suarez LD, Kapur S, Bielory L. Hereditary Angioedema and Gastrointestinal Complications: An Extensive Review of the Literature. Case Rep Immunol. 2015;2015:925861. doi: 10.1155/2015/925861"),
    (13, "Williams AH, Craig TJ. Perioperative management for patients with hereditary angioedema. Allergy Rhinol (Providence). 2015;6(1):50-55. doi: 10.2500/ar.2015.6.0112"),
    (14, "Wu MA, Zanichelli A, Mansi M, Cicardi M. Current treatment options for hereditary angioedema due to C1 inhibitor deficiency. Expert Opin Pharmacother. 2015. doi: 10.1517/14656566.2016.1104300"),
    (15, "Zanichelli A, Arcoleo F, Barca MP, Borrelli P, Bova M, Cancian M, et al. A nationwide survey of hereditary angioedema due to C1 inhibitor deficiency in Italy. Orphanet J Rare Dis. 2015;10:11. doi: 10.1186/s13023-015-0233-x"),
    (16, "Zanichelli A, Magerl M, Longhurst H, Aberer W, Bouillet L, Fabien V, et al. Efficacy of on-demand treatment in reducing morbidity in patients with hereditary angioedema due to C1-inhibitor deficiency: results from the Icatibant Outcome Survey. Allergy. 2015;70(2):218-22. doi: 10.1111/all.12555"),
    (17, "Zuraw BL, Cicardi M, Longhurst HJ, Bernstein JA, Li HH, Magerl M, et al. Phase II study results of a replacement therapy for hereditary angioedema with subcutaneous C1-inhibitor concentrate. Allergy. 2015;70:1319-1328. doi: 10.1111/all.12658"),
    (18, "Christiansen SC, Davis DK, Castaldo AJ, Zuraw BL. Pediatric Hereditary Angioedema: Onset, Diagnostic Delay, and Disease Severity. Clin Pediatr (Phila). 2016;55(10):935-42. doi: 10.1177/0009922815616886"),
    (19, "Frank MM, Zuraw B, Banerji A, Bernstein JA, Craig T, Busse P, et al. Management of Children With Hereditary Angioedema Due to C1 Inhibitor Deficiency. Pediatrics. 2016;138(5):e20160575. doi: 10.1542/peds.2016-0575"),
    (20, "Fu LW, Freedman-Kalchman T, Betschel S, Sussman G. Review of hereditary angioedema. LymphoSign Journal. 2016;3:47-53. doi: 10.14785/lymphosign-2016-0001"),
    (21, "Germenis AE, Speletas M. Genetics of Hereditary Angioedema Revisited. Clin Rev Allergy Immunol. 2016;51(2):170-82. doi: 10.1007/s12016-016-8543-x"),
    (22, "González-Quevedo T, Larco JI, Marcos C, Guilarte M, Baeza ML, Cimbollek S, et al. Management of Pregnancy and Delivery in Patients With Hereditary Angioedema Due to C1 Inhibitor Deficiency. J Investig Allergol Clin Immunol. 2016;26(3):161-7. doi: 10.18176/jiaci.0037"),
    (23, "Henao MP, Kraschnewski JL, Kelbel T, Craig TJ. Diagnosis and screening of patients with hereditary angioedema in primary care. Ther Clin Risk Manag. 2016;12:701-711. doi: 10.2147/TCRM.S86293"),
    (24, "Hofman ZLM, Relan A, Zeerleder S, Drouet C, Zuraw B, Hack CE. Angioedema attacks in patients with hereditary angioedema: Local manifestations of a systemic activation process. J Allergy Clin Immunol. 2016;138:359-66. doi: 10.1016/j.jaci.2016.02.041"),
    (25, "Loffredo S, Bova M, Suffritti C, Borriello F, Zanichelli A, Petraroli A, et al. Elevated plasma levels of vascular permeability factors in C1 inhibitor-deficient hereditary angioedema. Allergy. 2016;71(7):989-96. doi: 10.1111/all.12871"),
    (26, "Longhurst H, Bygum A. The Humanistic, Societal, and Pharmaco-economic Burden of Angioedema. Clin Rev Allerg Immunol. 2016;51:230-239. doi: 10.1007/s12016-016-8575-2"),
    (27, "Martinez-Saguer I, Farkas H. Erythema Marginatum as an Early Symptom of Hereditary Angioedema: Case Report of 2 Newborns. Pediatrics. 2016 Feb;137(2):e20152411. doi: 10.1542/peds.2015-2411"),
    (28, "Piñero-Saavedra M, González-Quevedo T, Saenz de San Pedro B, Alcaraz C, Bobadilla-González P, Fernández-Vieira L, et al. Hereditary angioedema with F12 mutation: Clinical features and enzyme polymorphisms in 9 Southwestern Spanish families. Ann Allergy Asthma Immunol. 2016;117(5):520-526. doi: 10.1016/j.anai.2016.09.001"),
    (29, "Steiner UC, Weber-Chrysochoou C, Helbling A, Scherer K, Grendelmeier PS, Wuillemin WA. Hereditary angioedema due to C1 inhibitor deficiency in Switzerland: clinical characteristics and therapeutic modalities within a cohort study. Orphanet J Rare Dis. 2016;11:43. doi: 10.1186/s13023-016-0423-1"),
    (30, "Caballero T, Aberer W, Longhurst HJ, Maurer M, Zanichelli A, Perrin A, et al. The Icatibant Outcome Survey: experience of hereditary angioedema management from six European countries. J Eur Acad Dermatol Venereol. 2017;31:1214-1222. doi: 10.1111/jdv.14251"),
    (31, "Farkas H, Martinez-Saguer I, Bork K, Bowen T, Craig T, Frank M, et al. International consensus on the diagnosis and management of pediatric patients with hereditary angioedema with C1 inhibitor deficiency. Allergy. 2017;72:300-313. doi: 10.1111/all.13001"),
    (32, "Gianni P, Loules G, Zamanakou M, Kompoti M, Csuka D, Psarros F, et al. Genetic Determinants of C1 Inhibitor Deficiency Angioedema Age of Onset. Int Arch Allergy Immunol. 2017;174(3-4):131-135. doi: 10.1159/000481987"),
    (33, "Kaplan AP, Maas C. The Search for Biomarkers in Hereditary Angioedema. Front Med. 2017;4:206. doi: 10.3389/fmed.2017.00206"),
    (34, "Magerl M, Frank M, Lumry W, et al. Short-term prophylactic use of C1-inhibitor concentrate in hereditary angioedema: Findings from an international patient registry. Ann Allergy Asthma Immunol. 2017;118(1):110-112. doi: 10.1016/j.anai.2016.10.006"),
    (35, "Maurer M, Magerl M, Ansotegui I, Aygören-Pürsün E, Betschel S, Bork K, et al. The international WAO/EAACI guideline for the management of hereditary angioedema - the 2017 revision and update. World Allergy Organ J. 2018;11(1):5. doi: 10.1186/s40413-017-0180-1"),
    (36, "Riedl MA, Grivcheva-Panovska V, Moldovan D, Baker J, Yang WH, Giannetti BM, et al. Recombinant human C1 esterase inhibitor for prophylaxis of hereditary angio-oedema: a phase 2, multicentre, randomised, double-blind, placebo-controlled crossover trial. Lancet. 2017;390(10102):1595-602. doi: 10.1016/S0140-6736(17)31963-3"),
    (37, "Bork K, Wulff K, Steinmüller-Magin L, Braenne I, Staubach-Renz P, Witzke G, et al. Hereditary angioedema with a mutation in the plasminogen gene. Allergy. 2018;73(2):442-50. doi: 10.1111/all.13270"),
    (38, "Craig T, Busse P, Gower RG, Johnston DT, Kashkin JM, Li HH, et al. Long-term prophylaxis therapy in patients with hereditary angioedema with C1 inhibitor deficiency. Ann Allergy Asthma Immunol. 2018;121(6):673-679. doi: 10.1016/j.anai.2018.07.025"),
    (39, "Dewald G. A missense mutation in the plasminogen gene, within the plasminogen kringle 3 domain, in hereditary angioedema with normal C1 inhibitor. Biochem Biophys Res Commun. 2018;498(1):193-198. doi: 10.1016/j.bbrc.2017.12.060"),
    (40, "Hakl R, Kuklínek P, Krčmová I, Králíčková P, Freiberger T, Janků P, et al. Treatment of Hereditary Angioedema Attacks with Icatibant and Recombinant C1 Inhibitor During Pregnancy. J Clin Immunol. 2018;38(7):810-5. doi: 10.1007/s10875-018-0553-4"),
    (41, "Loffredo S, Ferrara AL, Bova M, Borriello F, Suffritti C, Veszeli N, et al. Secreted Phospholipases A2 in Hereditary Angioedema With C1-Inhibitor Deficiency. Front Immunol. 2018;9:1721. doi: 10.3389/fimmu.2018.01721"),
    (42, "Loules G, Zamanakou M, Parsopoulou F, Vatsiou S, Psarros F, Csuka D, et al. Targeted next-generation sequencing for the molecular diagnosis of hereditary angioedema due to C1-inhibitor deficiency. Gene. 2018;667:76-82. doi: 10.1016/j.gene.2018.05.029"),
    (43, "Lumry WR. Hereditary Angioedema: The Economics of Treatment of an Orphan Disease. Front Med. 2018;5:22. doi: 10.3389/fmed.2018.00022"),
    (44, "van den Elzen M, Go MFCL, Knulst AC, Blankestijn MA, van Os-Medendorp H, Otten HG. Efficacy of Treatment of Non-hereditary Angioedema. Clin Rev Allerg Immunol. 2018;54:412-431. doi: 10.1007/s12016-016-8585-0"),
    (45, "Aygören-Pürsün E, Soteres DF, Nieto-Martinez SA, Christensen J, Jacobson KW, Moldovan D, et al. A randomized trial of human C1 inhibitor prophylaxis in children with hereditary angioedema. Pediatr Allergy Immunol. 2019;30:553-61. doi: 10.1111/pai.13060"),
    (46, "Betschel S, Badiou J, Binkley K, Borici-Mazi R, Hébert J, Kanani A, et al. The International/Canadian Hereditary Angioedema Guideline. Allergy Asthma Clin Immunol. 2019;15:72. doi: 10.1186/s13223-019-0376-8"),
    (47, "Hwang JR, Hwang G, Johri A, Craig T. Oral plasma kallikrein inhibitor BCX7353 for treatment of hereditary angioedema. Immunotherapy. 2019;11(17):1439-1444. doi: 10.2217/imt-2019-0128"),
    (48, "Levi M, Cohn DM, Zeerleder S. Hereditary angioedema: Linking complement regulation to the coagulation system. Res Pract Thromb Haemost. 2019;3(1):38-43. doi: 10.1002/rth2.12175"),
    (49, "Liu J, Qin J, Borodovsky A, Racie T, Castoreno A, Schlegel M, et al. An investigational RNAi therapeutic targeting Factor XII (ALN-F12) for the treatment of hereditary angioedema. RNA. 2019;25(2):255-63. doi: 10.1261/rna.068916.118"),
    (50, "Banerji A, Davis KH, Brown TM, Hollis K, Hunter SM, Long J, et al. Patient-reported burden of hereditary angioedema: findings from a patient survey in the United States. Ann Allergy Asthma Immunol. 2020;124(6):600-607. doi: 10.1016/j.anai.2020.02.018"),
    (51, "Bova M, Suffritti C, Bafunno V, Loffredo S, Cordisco G, Del Giacco S, et al. Impaired control of the contact system in hereditary angioedema with normal C1-inhibitor. Allergy. 2020;75(6):1394-1403. doi: 10.1111/all.14160"),
    (52, "Busse PJ, Christiansen SC, Riedl MA, Banerji A, Bernstein JA, Castaldo AJ, et al. US HAEA Medical Advisory Board 2020 Guidelines for the Management of Hereditary Angioedema. J Allergy Clin Immunol Pract. 2021;9(1):132-150.e3. doi: 10.1016/j.jaip.2020.08.046"),
    (53, "Cohn DM, Viney NJ, Fijen LM, Schneider E, Alexander VJ, Xia S, et al. Antisense Inhibition of Prekallikrein to Control Hereditary Angioedema. N Engl J Med. 2020 Sep 24;383(13):1242-1247. doi: 10.1056/NEJMoa1915035"),
    (54, "Craig T. Triggers and short-term prophylaxis in patients with hereditary angioedema. Allergy Asthma Proc. 2020;41(6):S30-S34. doi: 10.2500/aap.2020.41.200058"),
    (55, "Gompel A, Fain O, Boccon-Gibod I, Gobert D, Bouillet L. Exogenous Hormones and Hereditary angioedema. Int Immunopharmacol. 2020;78:106080. doi: 10.1016/j.intimp.2019.106080"),
    (56, "Hujová P, Souček P, Grodecká L, Grombiříková H, Ravčuková B, Kuklínek P, et al. Deep Intronic Mutation in SERPING1 Caused Hereditary Angioedema Through Pseudoexon Activation. J Clin Immunol. 2020;40(3):435-46. doi: 10.1007/s10875-020-00753-2"),
    (57, "Lumry WR, Settipane RA. Hereditary angioedema: Epidemiology and burden of disease. Allergy Asthma Proc. 2020;41(Suppl 1):S8-S13. doi: 10.2500/aap.2020.41.200050"),
    (58, "Proper SP, Lavery WJ, Bernstein JA. Definition and classification of hereditary angioedema. Allergy Asthma Proc. 2020;41(Suppl 1):S3-S7. doi: 10.2500/aap.2020.41.200040"),
    (59, "Vatsiou S, Zamanakou M, Loules G, Psarros F, Parsopoulou F, Csuka D, et al. A novel deep intronic SERPING1 variant as a cause of hereditary angioedema due to C1-inhibitor deficiency. Allergol Int. 2020;69:443-449. doi: 10.1016/j.alit.2019.12.009"),
    (60, "Zanichelli A, Ghezzi M, Santicchia I, Vacchini R, Cicardi M, Sparaco A, et al. Short-term prophylaxis in patients with angioedema due to C1-inhibitor deficiency undergoing dental procedures: An observational study. PLoS ONE. 2020;15(3):e0230128."),
    (61, "Bork K, Anderson JT, Caballero T, Craig T, Johnston DT, Li HH, et al. Assessment and management of disease burden and quality of life in patients with hereditary angioedema: a consensus report. Allergy Asthma Clin Immunol. 2021;17(1):40. doi: 10.1186/s13223-021-00537-2"),
    (62, "Caballero T. Treatment of Hereditary Angioedema. J Investig Allergol Clin Immunol. 2021;31(1):1-16. doi: 10.18176/jiaci.0653"),
    (63, "Fijen LM, Bork K, Cohn DM. Current and Prospective Targets of Pharmacologic Treatment of Hereditary Angioedema Types 1 and 2. Clin Rev Allergy Immunol. 2021;61:66-76. doi: 10.1007/s12016-021-08832-x"),
    (64, "Lesser H, Cohn JE. Hereditary angioedema. Int J Emerg Med. 2021;14:43. doi: 10.1186/s12245-021-00364-7"),
    (65, "Maurer M, Aygören-Pürsün E, Banerji A, Bernstein JA, Boysen HB, Busse PJ, et al. Consensus on treatment goals in hereditary angioedema: A global Delphi initiative. J Allergy Clin Immunol. 2021;148(6):1526-32. doi: 10.1016/j.jaci.2021.05.016"),
    (66, "Maurer M, Magerl M, Betschel S, et al. The international WAO/EAACI guideline for the management of hereditary angioedema - The 2021 revision and update. World Allergy Organ J. 2022;15(3):100627. doi: 10.1016/j.waojou.2022.100627"),
    (67, "Mendivil J, Murphy R, de la Cruz M, Janssen E, Boysen HB, Jain G, et al. Clinical characteristics and burden of illness in patients with hereditary angioedema: findings from a multinational patient survey. Orphanet J Rare Dis. 2021;16:94. doi: 10.1186/s13023-021-01717-4"),
    (68, "Ohsawa I, Honda D, Fukuda T, Kohga K, Morita E, Moriwaki S, et al. Oral berotralstat for the prophylaxis of hereditary angioedema attacks in patients in Japan: A phase 3 randomized trial. Allergy. 2021;76(6):1789-99. doi: 10.1111/all.14674"),
    (69, "Santacroce R, D'Andrea G, Maffione AB, Margaglione M, d'Apolito M. The Genetics of Hereditary Angioedema: A Review. J Clin Med. 2021;10(9):2023. doi: 10.3390/jcm10092023"),
    (70, "Zuraw B, Lumry WR, Johnston DT, Aygören-Pürsün E, Banerji A, Bernstein JA, et al. Oral once-daily berotralstat for the prevention of hereditary angioedema attacks: A randomized, double-blind, placebo-controlled phase 3 trial. J Allergy Clin Immunol. 2021;148:164-72. doi: 10.1016/j.jaci.2020.10.015"),
    (71, "ASCIA HAE Working Party. Hereditary Angioedema (HAE) Position Paper. Australasian Society of Clinical Immunology and Allergy; 2022."),
    (72, "Beard N, Frese M, Smertina E, Mere P, Katelaris C, Mills K. Interventions for the long-term prevention of hereditary angioedema attacks. Cochrane Database of Systematic Reviews. 2022;11:CD013403. doi: 10.1002/14651858.CD013403.pub2"),
    (73, "Fijen LM, Riedl MA, Bordone L, Bernstein JA, Raasch J, Tachdjian R, et al. Inhibition of Prekallikrein for Hereditary Angioedema. N Engl J Med. 2022 Mar 17;386(11):1026-1033. doi: 10.1056/NEJMoa2109329"),
    (74, "Guo Y, Zhang H, Lai H, Wang H, Chong-Neto HJ, Valle SOR, et al. Long-term Prophylaxis with Androgens in the management of Hereditary Angioedema (HAE) in emerging countries. Orphanet J Rare Dis. 2022;17:399. doi: 10.1186/s13023-022-02536-x"),
    (75, "Kesh S, Bernstein JA. Isolated angioedema: A review of classification and update on management. Ann Allergy Asthma Immunol. 2022;129(6):692-702. doi: 10.1016/j.anai.2022.08.003"),
    (76, "Maurer M, Aberer W, Caballero T, Bouillet L, Grumach AS, Botha J, et al. The Icatibant Outcome Survey: 10 years of experience with icatibant for patients with hereditary angioedema. Clin Exp Allergy. 2022;52(9):1048-1058. doi: 10.1111/cea.14206"),
    (77, "Sundler Björkman L, Persson B, Aronsson D, Skattum L, Nordenfelt P, Egesten A. Comorbidities in hereditary angioedema-A population-based cohort study. Clin Transl Allergy. 2022;12(4):e12135. doi: 10.1002/clt2.12135"),
    (78, "Valerieva A, Longhurst HJ. Treatment of hereditary angioedema-single or multiple pathways to the rescue. Front Allergy. 2022;3:952233. doi: 10.3389/falgy.2022.952233"),
    (79, "Betschel SD, Banerji A, Busse PJ, Cohn DM, Magerl M. Hereditary Angioedema: A Review of the Current and Evolving Treatment Landscape. J Allergy Clin Immunol Pract. 2023 Aug;11(8):2315-2325. doi: 10.1016/j.jaip.2023.04.017"),
    (80, "Branco Ferreira M, Baeza ML, Spinola Santos A, Prieto-Garcia A, Leal R, Alvarez J, et al. Evolution of Guidelines for the Management of Hereditary Angioedema due to C1 Inhibitor Deficiency. J Investig Allergol Clin Immunol. 2023;33(5):332-62. doi: 10.18176/jiaci.0909"),
    (81, "Caballero T, Lleonart-Bellfill R, Pedrosa M, Ferrer L, Guilarte M. Expert Review and Consensus on the Treat-to-Target Management of Hereditary Angioedema: From Scientific Evidence to Clinical Practice. J Investig Allergol Clin Immunol. 2023;33(4):238-49. doi: 10.18176/jiaci.0875"),
    (82, "Chong-Neto HJ. A narrative review of recent literature of the quality of life in hereditary angioedema patients. World Allergy Organ J. 2023;16(5):100758. doi: 10.1016/j.waojou.2023.100758"),
    (83, "Craig TJ, Reshef A, Li HH, Jacobs JS, Bernstein JA, Farkas H, et al. Efficacy and safety of garadacimab, a factor XIIa inhibitor for hereditary angioedema prevention (VANGUARD): a global, multicentre, randomised, double-blind, placebo-controlled, phase 3 trial. Lancet. 2023;401(10382):1079-1090. doi: 10.1016/S0140-6736(23)00350-1"),
    (84, "Diaz-Menindez M, Morgenstern-Kaplan D, Cuervo-Pardo L, Alvarez-Arango S, Gonzalez-Estrada A. Prevention of Recurrent Attacks of Hereditary Angioedema (HAE): Berotralstat and Its Oral Bioavailability. Ther Clin Risk Manag. 2023;19:313-317. doi: 10.2147/TCRM.S310376"),
    (85, "Grumach AS, Gadir N, Kessel A, Yegin A, Martinez-Saguer I, Bernstein JA. Current challenges and future opportunities in patient-focused management of hereditary angioedema: A narrative review. Clin Transl Allergy. 2023;13(5):e12243. doi: 10.1002/clt2.12243"),
    (86, "Johnson F, Stenzl A, Hofauer B, Heppt H, Ebert EV, Wollenberg B, et al. A Retrospective Analysis of Long-Term Prophylaxis with Berotralstat in Patients with Hereditary Angioedema and Acquired C1-Inhibitor Deficiency-Real-World Data. Clin Rev Allergy Immunol. 2023;65:354-364. doi: 10.1007/s12016-023-08972-2"),
    (87, "Jones D, Zafra H, Anderson J. Managing Diagnosis, Treatment, and Burden of Disease in Hereditary Angioedema Patients with Normal C1-Esterase Inhibitor. J Asthma Allergy. 2023;16:447-460. doi: 10.2147/JAA.S398333"),
    (88, "Longhurst H, Valerieva A. A Review of Randomized Controlled Trials of Hereditary Angioedema Long-Term Prophylaxis with C1 Inhibitor Replacement Therapy: Alleviation of Disease Symptoms Is Achievable. J Asthma Allergy. 2023;16:269-277. doi: 10.2147/JAA.S396338"),
    (89, "Maurer M, Abuzakouk M, Al-Ahmad M, Al-Herz W, Alrayes H, Al-Tamemi S, et al. Consensus on diagnosis and management of Hereditary Angioedema in the Middle East: A Delphi initiative. World Allergy Organ J. 2023 Jan;16(1):100729. doi: 10.1016/j.waojou.2022.100729"),
    (90, "Mormile I, Palestra F, Petraroli A, Loffredo S, Rossi FW, Spadaro G, et al. Neurologic and Psychiatric Manifestations of Bradykinin-Mediated Angioedema: Old and New Challenges. Int J Mol Sci. 2023;24(15):12184. doi: 10.3390/ijms241512184"),
    (91, "Raasch J, Glaum MC, O'Connor M. The multifactorial impact of receiving a hereditary angioedema diagnosis. World Allergy Organ J. 2023;16(6):100792. doi: 10.1016/j.waojou.2023.100792"),
    (92, "Rosa A, Franco R, Miranda M, Casella S, D'Amico C, Fiorillo L, et al. The role of anxiety in patients with hereditary angioedema during oral treatment: a narrative review. Front Oral Health. 2023;4:1257703. doi: 10.3389/froh.2023.1257703"),
    (93, "Shamanaev A, Dickeson SK, Ivanov I, Litvak M, Sun M-F, Kumar S, et al. Mechanisms involved in hereditary angioedema with normal C1-inhibitor activity. Front Physiol. 2023;14:1146834. doi: 10.3389/fphys.2023.1146834"),
    (94, "Sinnathamby ES, Issa PP, Roberts L, Norwood H, Malone K, Vemulapalli H, et al. Hereditary Angioedema: Diagnosis, Clinical Implications, and Pathophysiology. Adv Ther. 2023;40:814-827. doi: 10.1007/s12325-022-02401-0"),
    (95, "Tachdjian R, Kaplan AP. A Comprehensive Management Approach in Pediatric and Adolescent Patients With Hereditary Angioedema. Clin Pediatr (Phila). 2023;62(9):973-980. doi: 10.1177/00099228231155703"),
    (96, "Watt M, Malmenäs M, Romanus D, Haeussler K. Network meta-analysis for indirect comparison of lanadelumab and berotralstat for the treatment of hereditary angioedema. J Comp Eff Res. 2023;e220188. doi: 10.57264/cer-2022-0188"),
    (97, "Zuraw BL, Maurer M, Sexton DJ, Cicardi M. Therapeutic monoclonal antibodies with a focus on hereditary angioedema. Allergol Int. 2023;72:54-62. doi: 10.1016/j.alit.2022.06.001"),
    (98, "Brito-Robinson T, Ayinuola YA, Ploplis VA, Castellino FJ. Plasminogen missense variants and their involvement in cardiovascular and inflammatory disease. Front Cardiovasc Med. 2024;11:1406953. doi: 10.3389/fcvm.2024.1406953"),
    (99, "Cohn DM, Renné T. Targeting factor XIIa for therapeutic interference with hereditary angioedema. J Intern Med. 2024;296(4):311-26. doi: 10.1111/joim.20008"),
    (100,"Costanzo G, Sambugaro G, Firinu D. Hereditary angioedema due to C1-inhibitor deficiency: current therapeutic approaches. Curr Opin Allergy Clin Immunol. 2024;24(6):488-495. doi: 10.1097/ACI.0000000000001042"),
    (101,"Craig T, Tachdjian R, Bernstein JA, Anderson J, Nurse C, Watt M, et al. Long-term prevention of hereditary angioedema attacks with lanadelumab in adolescents. Ann Allergy Asthma Immunol. 2024;133(6):712-719. doi: 10.1016/j.anai.2024.08.001"),
    (102,"Ding L, Zhang MJ, Rao GW. Summary and future of medicine for hereditary angioedema. Drug Discov Today. 2024;29(3):103890. doi: 10.1016/j.drudis.2024.103890"),
    (103,"Fasshauer M, Wedi B. Hereditary angioedema (HAE) in children and adolescents: New treatment options. Allergol Select. 2024;8:336-345. doi: 10.5414/ALX02532E"),
    (104,"Giavina-Bianchi P, Aun MV, Giavina-Bianchi M, Ribeiro AJ, Agondi RC, Motta AA, et al. Hereditary angioedema classification: Expanding knowledge by genotyping and endotyping. World Allergy Organ J. 2024;17(5):100906. doi: 10.1016/j.waojou.2024.100906"),
    (105,"Jappe U, Bergmann KC, Brinkmann F, et al. Biologics in allergology and clinical immunology: Update on therapies for atopic diseases, urticaria, and angioedema and on safety aspects focusing on hypersensitivity reactions. Allergol Select. 2024;8:365-406. doi: 10.5414/ALX02533E"),
    (106,"Kiani-Alikhan S, Gower R, Craig T, Wedner HJ, Kinaciyan T, Aygören-Pürsün E, et al. Once-Daily Oral Berotralstat for Long-Term Prophylaxis of Hereditary Angioedema: The Open-Label Extension of the APEX-2 Randomized Trial. J Allergy Clin Immunol Pract. 2024;12(3):733-743.e10. doi: 10.1016/j.jaip.2023.12.019"),
    (107,"Long LH, Fujioka T, Craig TJ, Hitomi H. Long-term outcome of C1-esterase inhibitor deficiency. Asian Pac J Allergy Immunol. 2024;42(3):222-232. doi: 10.12932/ap-220224-1792"),
    (108,"Longhurst HJ, Lindsay K, Petersen RS, Fijen LM, Gurugama P, Maag D, et al. CRISPR-Cas9 In Vivo Gene Editing of KLKB1 for Hereditary Angioedema. N Engl J Med. 2024 Feb 1;390(5):432-441. doi: 10.1056/NEJMoa2309149"),
    (109,"Pagnier A, Dermesropian A, Kevorkian-Verguet C, Bourgoin-Heck M, Hoarau C, Reumaux H, et al. Hereditary angioedema in children: Review and practical perspective for clinical management. Pediatr Allergy Immunol. 2024;35(5):e14268. doi: 10.1111/pai.14268"),
    (110,"Petersen RS, Bordone L, Riedl MA, Tachdjian R, Craig TJ, Lumry WR, et al. A phase 2 open-label extension study of prekallikrein inhibition with donidalorsen for hereditary angioedema. Allergy. 2024;79:724-734. doi: 10.1111/all.15948"),
    (111,"Petersen RS, Fijen LM, Cohn DM. Efficacy outcomes in trials with prophylactic hereditary angioedema therapy: A systematic review. Allergy. 2024. doi: 10.1111/all.15962"),
    (112,"Petersen RS, Fijen LM, Kelder JP, Cohn DM. Deucrictibant for angioedema due to acquired C1-inhibitor deficiency: A randomized-controlled trial. J Allergy Clin Immunol. 2024;154:179-83. doi: 10.1016/j.jaci.2024.03.007"),
    (113,"Petersen RS, Fijen LM, Levi M, Cohn DM. Hereditary Angioedema: The Clinical Picture of Excessive Contact Activation. Semin Thromb Hemost. 2024;50(7):978-988. doi: 10.1055/s-0042-1758820"),
    (114,"Radojicic C, Anderson J. Hereditary angioedema with normal C1 esterase inhibitor: Current paradigms and clinical dilemmas. Allergy Asthma Proc. 2024;45(3):147-157. doi: 10.2500/aap.2024.45.240010"),
    (115,"Raja A, Shuja MH, Raja S, Qammar A, Kumar S, Khurram L, et al. Efficacy and safety of Donidalorsen in Hereditary Angioedema with C1 inhibitor deficiency: a systematic review and a meta analysis. Arch Dermatol Res. 2024. doi: 10.1007/s00403-024-03652-3"),
    (116,"Smith TD, Riedl MA. The future of therapeutic options for hereditary angioedema. Ann Allergy Asthma Immunol. 2024;133(4):380-390. doi: 10.1016/j.anai.2024.04.029"),
    (117,"Soteres D, Lumry W, Magerl M, Gagnon R, Desai B, Tomita D, et al. Berotralstat improved quality of life through 96 weeks across multiple subgroups of patients with hereditary angioedema. Allergy. 2024;80:2361-70. doi: 10.1111/all.16109"),
    (118,"Tutunaru CV, Ică OM, Mitroi GG, Neagoe CD, Mitroi GF, Orzan OA, et al. Unveiling the Complexities of Hereditary Angioedema. Biomolecules. 2024;14(10):1298. doi: 10.3390/biom14101298"),
    (119,"Valerieva A, Caballero T, Magerl M, Frade JP, Audhya PK, Craig T. Advent of oral medications for the treatment of hereditary angioedema. Clin Transl Allergy. 2024;14(9):e12391. doi: 10.1002/clt2.12391"),
    (120,"Wisniewski P, Gangnus T, Burckhardt BB. Recent advances in the discovery and development of drugs targeting the kallikrein-kinin system. J Transl Med. 2024;22:388. doi: 10.1186/s12967-024-05216-5"),
    (121,"Baroni I, Paglione G, De Angeli G, Angolani M, Callus E, Magon A, et al. A COSMIN systematic review of instruments for evaluating health-related quality of life in people with Hereditary Angioedema. Health Qual Life Outcomes. 2025;23(1):12. doi: 10.1186/s12955-025-02342-6"),
    (122,"Christiansen SC, Banerji A, Bernstein JA, Busse PJ, Craig T, Li HH, et al. Hereditary Angioedema With Normal C1 Inhibitor: A Quarter Century of Forward Progress and Persisting Obstacles. J Allergy Clin Immunol Pract. 2025;13(6):1300-9. doi: 10.1016/j.jaip.2025.02.036"),
    (123,"Cohn DM, Soteres DF, Craig TJ, Lumry WR, Magerl M, Riedl MA, et al. Interplay between on-demand treatment trials for hereditary angioedema and treatment guidelines. J Allergy Clin Immunol. 2025;155:726-39. doi: 10.1016/j.jaci.2024.12.1079"),
    (124,"Fung S. Garadacimab: First Approval. Drugs. 2025;85:827-832. doi: 10.1007/s40265-025-02180-2"),
    (125,"Gao H, Zhao Y, Chen S, Zhang Z, Yang F, Chen Z, et al. Expanding the Genetic and Clinical Spectrum of Hereditary Angioedema with Normal C1 Inhibitor: Novel Variants and Treatment Insights. J Clin Immunol. 2025;45:124. doi: 10.1007/s10875-025-01912-z"),
    (126,"Germenis AE, Sanoudou D. Incidental findings related to genes associated to HAE-nC1INH: how to proceed? Front Immunol. 2025;16:1605727. doi: 10.3389/fimmu.2025.1605727"),
    (127,"Ghosh D, Anderson J, Singh U, Bernstein CK, Bernstein JA. Clinical response and corresponding blood transcriptome pathways before and after treatment of hereditary angioedema prodromes compared to active swelling attacks. J Allergy Clin Immunol. 2025;155:947-55. doi: 10.1016/j.jaci.2024.11.035"),
    (128,"Grumach AS, Riedl MA, Cheng L, Jain S, Estepan DN, Zanichelli A. Hereditary angioedema diagnosis: Reflecting on the past, envisioning the future. World Allergy Organ J. 2025;18(6):101060. doi: 10.1016/j.waojou.2025.101060"),
    (129,"Riedl MA, Staubach P, Farkas H, Zanichelli A, Ren H, Nurse C, et al. Lanadelumab for prevention of attacks of non-histaminergic normal C1 inhibitor angioedema: results from the randomized, double-blind CASPIAN Study and CASPIAN open-label extension. Front Immunol. 2025;16:1502325. doi: 10.3389/fimmu.2025.1502325"),
    (130,"Walsh S, Bartlett M, Salvo-Halloran EM, Sears J, Li Y, Kelly M, et al. Network Meta-Analysis of Pharmacological Therapies for Long-Term Prophylactic Treatment of Patients with Hereditary Angioedema. Drugs R D. 2025;25:161-178. doi: 10.1007/s40268-025-00511-y"),
    (131,"Walsh S, Haltner A, Bartlett M, Sears J, Li Y, Kelly M, et al. Matching-adjusted indirect comparison between garadacimab and lanadelumab for the long-term prophylactic treatment of patients with hereditary angioedema. J Comp Eff Res. 2025. doi: 10.57264/cer-2024-0237"),
    (132,"Watt M, Goldgrub R, Malmenäs M, Haeussler K. Indirect treatment comparison of lanadelumab and a C1-esterase inhibitor in pediatric patients with hereditary angioedema. J Comp Eff Res. 2025;e240110. doi: 10.57264/cer-2024-0110"),
    (133,"Zanichelli A, De Angeli G, Baroni I, Mansi M, Caravella G, Caruso R. Hereditary angioedema treatment beyond biologics: current state of preventive and on-demand approaches and new perspectives. Expert Opin Pharmacother. 2025;26(10):1221-1228. doi: 10.1080/14656566.2025.2509782"),
    (134,"Zuraw BL, Bork K, Bouillet L, Christiansen SC, Farkas H, Germenis AE, et al. Hereditary Angioedema with Normal C1 Inhibitor: an Updated International Consensus Paper on Diagnosis, Pathophysiology, and Treatment. Clin Rev Allergy Immunol. 2025. doi: 10.1007/s12016-025-09027-4"),
]


def parse_citation(no, text):
    doi_match = re.search(r'doi:\s*(10\.\S+)', text, re.IGNORECASE)
    doi = doi_match.group(1).rstrip('.') if doi_match else ""

    year_match = re.search(r'\b(20\d{2}|199\d)\b', text)
    year = year_match.group(1) if year_match else ""

    # first_author: first word(s) before first comma
    author_match = re.match(r'^([A-ZÀ-Öa-zà-ö][^\s,]+(?:\s+[A-ZÀ-Ö]+)?)', text)
    first_author = author_match.group(1) if author_match else text.split(',')[0].strip()
    # Handle institutional authors like "ASCIA HAE Working Party"
    if first_author in ("ASCIA", "ASCIA HAE Working Party"):
        first_author = "ASCIA HAE Working Party"

    # title: extract the sentence after the author block ending with "et al. " or last initials
    title = ""
    after_authors = re.split(r'et al\.\s+', text, maxsplit=1)
    if len(after_authors) == 2:
        rest = after_authors[1]
    else:
        # no "et al." — authors end after last ". " that starts a title-like sentence
        parts = text.split('. ')
        if len(parts) >= 2:
            # find where author list ends: first part with "." at sentence boundary
            rest = '. '.join(parts[1:])
        else:
            rest = text

    # title is first sentence of rest
    title_match = re.match(r'^([^.]+(?:\.[^.A-Z][^.]*)?)\.\s+', rest)
    if title_match:
        title = title_match.group(1).strip()
    else:
        title = rest.split('.')[0].strip()

    # journal: text after title, before year/volume
    journal = ""
    after_title = rest[len(title):].lstrip('. ')
    jrnl_match = re.match(r'^([^.;0-9]+?)[\s.;]', after_title)
    if jrnl_match:
        journal = jrnl_match.group(1).strip().rstrip('.')

    # pubmed link
    if doi:
        pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/?term={doi}%5Bdoi%5D"
    else:
        q = f"{first_author}+{year}+hereditary+angioedema"
        pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/?term={q}"

    return {
        "citation_no": no,
        "first_author": first_author,
        "year": year,
        "title": title,
        "journal": journal,
        "doi": doi,
        "pubmed_link": pubmed_link,
        "full_citation": text,
    }


# ---------------------------------------------------------------------------
# Parse all 134
# ---------------------------------------------------------------------------
citations = {no: parse_citation(no, text) for no, text in CITATIONS_RAW}

# ---------------------------------------------------------------------------
# DOI → doc_id mapping (derived from filenames in corpus_manifest.csv)
# ---------------------------------------------------------------------------
DOI_TO_DOCID = {
    "10.1007/s12016-025-09027-4":       "HAE_061",
    "10.1016/j.jaci.2024.12.1079":      "HAE_062",
    "10.1186/s13223-021-00537-2":       "HAE_065",
    "10.1186/s13223-019-0376-8":        "HAE_066",
    "10.1186/s40413-017-0180-1":        "HAE_067",
    "10.1007/s40268-025-00511-y":       "HAE_070",
    "10.1186/s12955-025-02342-6":       "HAE_071",
    "10.1016/j.jaip.2023.04.017":       "HAE_074",
    "10.3389/froh.2023.1257703":        "HAE_076",
    "10.3389/fimmu.2025.1605727":       "HAE_085",
    "10.3389/fimmu.2025.1502325":       "HAE_086",
    "10.1016/j.waojou.2025.101060":     "HAE_087",
    "10.1007/s10875-025-01912-z":       "HAE_088",
    "10.12932/ap-220224-1792":          "HAE_089",
    "10.1186/s12967-024-05216-5":       "HAE_091",
    "10.3389/fcvm.2024.1406953":        "HAE_095",
    "10.1016/j.waojou.2024.100906":     "HAE_096",
    "10.1007/s12016-023-08972-2":       "HAE_098",
    "10.3389/fphys.2023.1146834":       "HAE_100",
    "10.1016/S0140-6736(23)00350-1":    "HAE_103",
    "10.1007/s12325-022-02401-0":       "HAE_104",
    "10.1186/s13023-022-02536-x":       "HAE_105",
    "10.1111/cea.14206":                "HAE_107",
    "10.3389/falgy.2022.952233":        "HAE_108",
    "10.1111/all.14674":                "HAE_109",
    "10.1007/s12016-021-08832-x":       "HAE_110",
    "10.1186/s13023-021-01717-4":       "HAE_111",
    "10.1186/s12245-021-00364-7":       "HAE_112",
    "10.1111/all.14160":                "HAE_113",
    "10.3389/fmed.2018.00022":          "HAE_117",
    "10.1007/s12016-016-8585-0":        "HAE_119",
    "10.2500/ar.2015.6.0112":           "HAE_120",
    "10.3389/fmed.2017.00206":          "HAE_122",
    "10.1186/s13023-016-0423-1":        "HAE_123",
    "10.1371/journal.pone.0142174":     "HAE_126",
    "10.1186/s13023-015-0233-x":        "HAE_127",
    "10.1007/s00403-024-03652-3":       "HAE_128",
    "10.1007/s10875-018-0553-4":        "HAE_129",
    "10.1007/s10875-020-00753-2":       "HAE_130",
    "10.1007/s40265-025-02180-2":       "HAE_132",
    "10.1016/S0140-6736(17)31963-3":    "HAE_029",
    "10.1056/NEJMoa1915035":            "HAE_023",
    "10.1056/NEJMoa2109329":            "HAE_024",
    "10.1056/NEJMoa2309149":            "HAE_025",
    "10.1172/JCI77139":                 "HAE_021",
    "10.1016/j.jaci.2021.05.016":       "HAE_026",
    "10.1016/j.jaci.2020.10.015":       "HAE_027",
    "10.1016/j.jaci.2024.11.035":       "HAE_028",
    "10.1111/joim.20008":               "HAE_022",
    "10.1016/j.jaci.2016.02.041":       "HAE_049",  # Hofman 2016
    "10.1016/j.anai.2016.09.001":       "HAE_007",  # Piñero-Saavedra 2016
    "10.1160/TH12-08-0549":             "HAE_008",  # Björkqvist 2013
    "10.18176/jiaci.0037":              "HAE_009",  # González-Quevedo 2016
    "10.1016/j.anai.2012.10.008":       "HAE_030",  # Lang 2012, AAAI (PIIS1081120612008125)
}

# Author+year → doc_id for named files
AUTHOR_YEAR_TO_DOCID = {
    ("Bernstein", "2015"): "HAE_040",
    ("Bork",      "2015"): "HAE_041",
    ("Christiansen","2016"): "HAE_042",
    ("Craig",     "2020"): "HAE_043",
    ("Frank",     "2016"): "HAE_045",
    ("Fu",        "2016"): "HAE_046",
    ("Germenis",  "2016"): "HAE_047",
    ("Gianni",    "2017"): "HAE_048",
    ("Hofman",    "2016"): "HAE_049",
    ("Hwang",     "2019"): "HAE_050",
    ("Longhurst", "2016"): "HAE_052",
    ("Lumry",     "2020"): "HAE_053",
    ("Martinez-Saguer","2016"): "HAE_054",
    ("Moldovan",  "2015"): "HAE_056",
    ("Proper",    "2020"): "HAE_060",
    ("Wu",        "2015"): "HAE_137",
    ("Zuraw",     "2013"): "HAE_138",
    ("Zuraw",     "2021"): "HAE_139",
}

# Other specific matches (overrides; applied last)
SPECIFIC_MATCHES = {
    # JIACI / special journal matches
    84:  "HAE_133",  # tcrm-19-313 → Diaz-Menindez 2023, TCRM 19:313
    62:  "HAE_134",  # vol31issue1_2 → Caballero 2021, JIACI 31(1):1-16
    81:  "HAE_135",  # vol33issue4_1 → Caballero 2023, JIACI 33(4):238
    80:  "HAE_136",  # vol33issue5_2 → Branco Ferreira 2023, JIACI 33(5):332
    87:  "HAE_051",  # jaa-16-447 → Jones 2023, J Asthma Allergy 16:447
    89:  "HAE_055",  # middle-east-Aadelphi-initiative → Maurer 2023, WAO J
    72:  "HAE_018",  # CD013403 → Beard 2022, Cochrane
    71:  "HAE_011",  # ASCIA_HP_Position_Paper_HAE_2022 → ASCIA 2022
    60:  "HAE_059",  # pone.0230128 → Zanichelli 2020, PLoS ONE (no doi in text)

    # ScienceDirect PIIS → citation (ISSN decoded)
    39:  "HAE_001",  # S0006-291X(17) → BBRC 2017/18 → Dewald 2018
    42:  "HAE_002",  # S0378-1119(18) → Gene 2018 → Loules 2018
    59:  "HAE_003",  # S1323-8930(20) → Allergol Int 2020 → Vatsiou 2020
    97:  "HAE_004",  # S1323-8930(22) → Allergol Int 2022 → Zuraw 2023
    102: "HAE_005",  # S1359-6446(24) → Drug Discov Today 2024 → Ding 2024
    55:  "HAE_006",  # S1567-5769(19) → Int Immunopharmacol 2020 → Gompel 2020
    34:  "HAE_031",  # PIIS1081120616311978 → AAAI 2016 → Magerl 2017
    38:  "HAE_032",  # PIIS1081120618306112 → AAAI 2018 → Craig 2018
    50:  "HAE_033",  # PIIS1081120620301459 → AAAI 2020 → Banerji 2020
    75:  "HAE_034",  # PIIS1081120622006603 → AAAI 2022 → Kesh 2022
    101: "HAE_035",  # PIIS1081120624002758 → AAAI 2024 → Craig 2024
    116: "HAE_036",  # PIIS1081120624004897 → AAAI 2024 → Smith 2024
    52:  "HAE_037",  # PIIS2213219820308783 → JAIP 2020 → Busse 2021
    106: "HAE_038",  # PIIS2213219823013685 → JAIP 2023 → Kiani-Alikhan 2024
    122: "HAE_039",  # PIIS2213219825002090 → JAIP 2025 → Christiansen 2025

    # Named-file matches
    10:  "HAE_013",  # Allergy-2014-Joseph → Joseph 2015, Allergy
    37:  "HAE_014",  # Allergy-2017-Bork-plasminogen → Bork 2018, Allergy
    110: "HAE_015",  # Allergy-2023-Petersen-donidalorsen → Petersen 2024
    111: "HAE_016",  # Allergy-2023-Petersen-efficacy → Petersen 2024
    1:   "HAE_019",  # DGA_2_WebseiteLeitl → Bork 2011 German Guideline
    133: "HAE_020",  # treatment beyond biologics → Zanichelli 2025
}

# ---------------------------------------------------------------------------
# Build citation_no → doc_id lookup
# ---------------------------------------------------------------------------
cit_to_docid = {}

# 1. DOI matching
for no, cdata in citations.items():
    doi = cdata["doi"]
    if doi and doi in DOI_TO_DOCID:
        cit_to_docid[no] = DOI_TO_DOCID[doi]

# 2. Author+year matching
for no, cdata in citations.items():
    if no in cit_to_docid:
        continue
    key = (cdata["first_author"].split()[0].rstrip(','), cdata["year"])
    if key in AUTHOR_YEAR_TO_DOCID:
        cit_to_docid[no] = AUTHOR_YEAR_TO_DOCID[key]

# 3. Specific overrides
for no, docid in SPECIFIC_MATCHES.items():
    cit_to_docid[no] = docid

# ---------------------------------------------------------------------------
# Read corpus_manifest.csv
# ---------------------------------------------------------------------------
manifest_path = os.path.join(os.path.dirname(__file__), "corpus_manifest.csv")
manifest = {}
with open(manifest_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        manifest[row['doc_id']] = row

# ---------------------------------------------------------------------------
# Build output rows
# ---------------------------------------------------------------------------
rows = []

# Track which doc_ids have been assigned
assigned_docids = set(cit_to_docid.values())

# Rows for the 134 citations
for no in range(1, 135):
    cdata = citations[no]
    docid = cit_to_docid.get(no, "")
    mrow = manifest.get(docid, {})
    rows.append({
        "doc_id":        docid,
        "filename":      mrow.get("filename", ""),
        "corpus_role":   mrow.get("corpus_role", "haystack") if docid else "haystack",
        "citation_no":   no,
        "first_author":  cdata["first_author"],
        "year":          cdata["year"],
        "title":         cdata["title"],
        "journal":       cdata["journal"],
        "doi":           cdata["doi"],
        "pubmed_link":   cdata["pubmed_link"],
        "full_citation": cdata["full_citation"],
    })

# Rows for doc_ids not matched to any citation (extra HAE files + HOLDOUT + DISTRACTOR)
for docid, mrow in sorted(manifest.items()):
    if docid in assigned_docids:
        continue
    rows.append({
        "doc_id":        docid,
        "filename":      mrow["filename"],
        "corpus_role":   mrow["corpus_role"],
        "citation_no":   "",
        "first_author":  "",
        "year":          "",
        "title":         "",
        "journal":       "",
        "doi":           "",
        "pubmed_link":   "",
        "full_citation": "",
    })

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
out_path = os.path.join(os.path.dirname(__file__), "corpus_list.csv")
fieldnames = [
    "doc_id", "filename", "corpus_role",
    "citation_no", "first_author", "year", "title", "journal",
    "doi", "pubmed_link", "full_citation",
]
with open(out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Written: {out_path}")
matched = sum(1 for r in rows if r["doc_id"] and r["citation_no"])
print(f"Matched citations->doc_id: {matched}/134")
unmatched_cits = [r["citation_no"] for r in rows if not r["doc_id"] and r["citation_no"]]
print(f"Citations without doc_id ({len(unmatched_cits)}): {unmatched_cits}")
unmatched_docs = [r["doc_id"] for r in rows if not r["citation_no"] and r["doc_id"]]
print(f"Doc_ids without citation ({len(unmatched_docs)}): {unmatched_docs}")
