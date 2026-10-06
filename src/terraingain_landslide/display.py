"""Draw stored endpoint values and intervals without recomputing them."""
from pathlib import Path
import csv, json

def render(tables, output, qa=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    OUT=Path(output); OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    QA=Path(qa) if qa else OUT/'display_checks'
    QA.mkdir(parents=True,exist_ok=True)
    tables=Path(tables)
    def audit_alignment(fig, path):
        try:
            from audit_panel_alignment import require_matplotlib_panel_alignment
        except ImportError:
            return
        require_matplotlib_panel_alignment(fig,json_out=path,tolerance_pt=1.5)
    plt.rcParams.update({'font.family':'Arial', 'font.size':8, 'axes.labelsize':8,
        'xtick.labelsize':7.5, 'ytick.labelsize':8, 'axes.titlesize':8,
        'legend.fontsize':7.5, 'axes.linewidth':0.6, 'pdf.fonttype':42,
        'ps.fonttype':42, 'svg.fonttype':'none', 'savefig.facecolor':'white'})
    TEAL='#187A85'; RUST='#BC4C37'; INK='#202020'; GREY='#B3B3B3'
    REGIONS=['dominicamaria','italy','hiroshima','hokkaido','thrissur']
    NAMES=dict(zip(REGIONS,['Dominica Maria','Italy','Hiroshima','Hokkaido','Thrissur']))
    RECIPES=['R0','R1','source_selected']
    LABELS={'R0':'R0 · lower encoder rate','R1':'R1 · higher encoder rate','source_selected':'Source recipe choice'}

    def rows(name):
        with (tables/name).open(encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f))

    def export(fig,name):
        fig.savefig(OUT/'figures'/f'{name}.pdf', metadata={'CreationDate':None,'ModDate':None,'Creator':'Matplotlib','Title':name})
        fig.savefig(OUT/'figures'/f'{name}.svg', metadata={'Date':None,'Creator':'Matplotlib','Title':name})
        fig.savefig(OUT/'figures'/f'{name}.png',dpi=600)
        fig.savefig(QA/f'{name}_preview.png',dpi=300)
        plt.close(fig)

    primary=rows('Table1_primary_endpoints.csv')
    paired=[r for r in rows('S2_outer_all_region_seed_endpoints.csv') if r['endpoint']=='20000']
    regional=rows('Table2_region_results.csv')
    assert len(primary)==3 and len(paired)==45 and len(regional)==15
    assert {(r['recipe'],r['region'],r['seed']) for r in paired} == {(a,b,str(c)) for a in RECIPES for b in REGIONS for c in [17,29,43]}

    # Figure 1: the two evaluation supports for the same specified procedures.
    fig,ax=plt.subplots(figsize=(7.2047244,3.5826772))
    fig.subplots_adjust(left=.28,right=.97,bottom=.23,top=.80)
    for i,r in enumerate(primary):
        y=2-i; inner=float(r['inner_gain_pp']); outer=float(r['outer_gain_pp'])
        lo=float(r['outer_ci95_low_pp']); hi=float(r['outer_ci95_high_pp'])
        ax.plot([outer,inner],[y,y],color='#D3D3D3',lw=.8,zorder=1)
        ax.errorbar(outer,y,xerr=[[outer-lo],[hi-outer]],fmt='s',ms=5,
            color=RUST,ecolor=RUST,elinewidth=1,capsize=2,zorder=3)
        ax.plot(inner,y,'o',ms=5,color=TEAL,zorder=3)
        ax.annotate(f'{outer:+.3f}',(outer,y),xytext=(0,-17),textcoords='offset points',ha='center',color=RUST,fontsize=7.5)
        ax.annotate(f'{inner:+.3f}',(inner,y),xytext=(0,-17),textcoords='offset points',ha='center',color=TEAL,fontsize=7.5)
    ax.axvline(0,color=INK,lw=.65,ls=(0,(3,3)))
    ax.set_yticks([2,1,0],[LABELS[x] for x in RECIPES])
    ax.set_xlim(-10,7); ax.set_ylim(-.6,2.55)
    ax.set_xticks([-10,-5,0,5]); ax.set_xlabel('Pretrained minus scratch F1 (percentage points)',labelpad=9)
    ax.tick_params(axis='y',length=0,pad=8)
    for s in ['top','right','left']:ax.spines[s].set_visible(False)
    fig.legend(handles=[Line2D([],[],marker='o',linestyle='none',color=TEAL,label='Source validation'),
        Line2D([],[],marker='s',linestyle='none',color=RUST,label='Complete regional holdout')],
        loc='upper center',bbox_to_anchor=(.64,.985),ncol=2,frameon=False)
    export(fig,'Figure1_source_and_holdout')

    # Figure 2: all 45 paired seed gains and all 15 stored regional means.
    fig,axes=plt.subplots(1,3,figsize=(7.2047244,4.1732283),sharey=True)
    fig.subplots_adjust(left=.19,right=.98,bottom=.22,top=.80,wspace=.13)
    seed_styles={'17':('o',TEAL,-.30),'29':('D','#42658E',-.10),'43':('s',RUST,.10)}
    for j,(ax,recipe) in enumerate(zip(axes,RECIPES)):
        for i,reg in enumerate(REGIONS):
            y=4-i
            ax.axhline(y,color='#EEEEEE',lw=.5,zorder=0)
            for seed,(marker,color,offset) in seed_styles.items():
                r=next(x for x in paired if (x['recipe'],x['region'],x['seed'])==(recipe,reg,seed))
                ax.plot(float(r['gain'])*100,y+offset,marker=marker,ms=4.5,
                    color=color,linestyle='none',zorder=3)
            m=next(x for x in regional if (x['recipe'],x['region'])==(recipe,reg))
            ax.plot(float(m['gain_pp']),y+.30,marker='o',ms=7,mfc='white',mec=INK,mew=1.1,linestyle='none',zorder=4)
        ax.axvline(0,color=INK,lw=.65,ls=(0,(3,3)))
        ax.set_xlim(-23,12);ax.set_ylim(-.6,4.6);ax.set_xticks([-20,-10,0,10])
        ax.set_title(['R0','R1','Source recipe choice'][j],pad=12)
        ax.annotate(chr(97+j),xy=(0,1),xycoords='axes fraction',xytext=(-7,14),textcoords='offset points',weight='bold',fontsize=9,annotation_clip=False)
        ax.tick_params(axis='y',length=0,pad=8)
        for s in ['top','right','left']:ax.spines[s].set_visible(False)
    axes[0].set_yticks([4,3,2,1,0],[NAMES[x] for x in REGIONS])
    fig.supxlabel('Held-region pretrained minus scratch F1 (percentage points)',y=.085,fontsize=8)
    handles=[Line2D([],[],marker=v[0],color=v[1],linestyle='none',ms=4.5,label='Seed '+k) for k,v in seed_styles.items()]
    handles.append(Line2D([],[],marker='o',color=INK,mfc='white',linestyle='none',ms=7,label='Three-seed mean'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.60,.99),ncol=4,frameon=False,columnspacing=1.25)
    fig.canvas.draw()
    audit_alignment(fig, QA/'panel_alignment.json')
    for r in paired:
        assert -23 < float(r['gain'])*100 < 12, 'A seed point is outside the plotted axis'
    export(fig,'Figure2_all_regions_and_seeds')

    # Figure S1: existing stored endpoints only; lines connect recorded checkpoints.
    trajectory=rows('S4_outer_macro_all_endpoints.csv')
    fig,ax=plt.subplots(figsize=(7.2047244,3.4251969))
    fig.subplots_adjust(left=.12,right=.96,bottom=.24,top=.85)
    for recipe,color,marker in [('R0',TEAL,'o'),('R1',RUST,'s')]:
        fixed=[r for r in trajectory if r['recipe']==recipe and r['endpoint'].isdigit()]
        ax.plot([int(r['endpoint'])/1000 for r in fixed],[float(r['gain'])*100 for r in fixed],
            marker=marker,ms=4.5,color=color,lw=1,label=recipe)
        selected=next(r for r in trajectory if r['recipe']==recipe and r['endpoint']=='inner_selected')
        ax.plot(26,float(selected['gain'])*100,marker=marker,ms=5,color=color,linestyle='none')
    ax.axvline(23,color='#C8C8C8',lw=.6)
    ax.axhline(0,color=INK,lw=.65,ls=(0,(3,3)))
    ax.set_ylim(-12.5,1); ax.set_xlim(4,28)
    ax.set_xticks([5,10,15,20,26],['5k','10k','15k','20k','Source checkpoint\nchoice'])
    ax.set_yticks([-12,-8,-4,0]);ax.set_ylabel('Average holdout $G$ (percentage points)')
    ax.set_xlabel('Attempted updates or separately selected checkpoint',labelpad=8)
    for s in ['top','right']:ax.spines[s].set_visible(False)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.19),ncol=2,frameon=False)
    export(fig,'FigureS1_existing_checkpoint_trajectory')
