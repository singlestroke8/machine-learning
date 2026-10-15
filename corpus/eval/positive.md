# 評価用: 正答すべき質問（言い換えテスト）

想定問答コーパスには含まれない言い回しで、対応するエントリIDに到達できるかを測る。
形式: `Q: 質問文 => 期待するエントリID`

Q: 六十歳から始めるのは遅すぎますか？ => age
Q: 定年後の趣味として考えているのですが => age
Q: 会社帰りに寄っても荷物になりませんか？ => no-instrument-needed
Q: 三味線って高いんでしょう？買わないとだめ？ => no-instrument-needed
Q: 家でも弾きたい場合はどうすればいい？ => rental
Q: 毎月の出費はトータルでどれくらい見ておけばいい？ => fee-overview
Q: 表示料金以外に後から請求されることはない？ => no-extra-fees
Q: PayPayやカードで月謝を払えますか？ => payment
Q: お金の支払いはどのタイミングですか？ => payment
Q: ピアノも挫折した私でも弾けるようになりますか？ => beginner
Q: まったくの素人なんですが恥ずかしくないですか？ => beginner
Q: 先生と一対一で教えてもらえるのですか？ => lesson-style
Q: 周りのレベルについていけるか心配です => lesson-style
Q: お稽古って怖い先生が多いイメージですが実際どうですか？ => strict
Q: ドレミが分からなくても平気？ => score
Q: 最終的にじょんから節が弾けるようになりますか？ => first-song
Q: 平日の夜にレッスンはやっていますか？ => schedule
Q: 1回のレッスンは何分ですか？ => schedule
Q: 教室への行き方を教えて => location
Q: 新大久保からも歩けますか？ => location
Q: 何区にあるんですか？ => address
Q: レッスン場所は音を出しても大丈夫なところ？ => dedicated-studio
Q: とりあえず一回だけ試してみたい => trial-lesson
Q: 見学や体験の申し込みはどこから？ => trial-flow
Q: 体験に行くと勧誘されそうで不安 => no-solicitation
Q: 合わなかったらすぐやめられますか？ => quit
Q: 発表の場はありますか？ => events
Q: ステージに立つのは強制ですか？ => recital-optional
Q: どんな先生方が在籍していますか？ => instructors
Q: 教室は何年くらい続いていますか？ => founder
Q: 有名な演奏家は出身にいますか？ => kondo
Q: LINEから申し込みたい => line
Q: 電話で聞きたいことがあるのですが => phone
