/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.nlpcraft.examples.sql.db

import java.util
import java.util.Collections

import com.typesafe.scalalogging.LazyLogging
import org.nlpcraft.model._
import org.nlpcraft.model.tools.sqlgen.NCSqlQuery

import scala.collection.JavaConverters._
import scala.language.implicitConversions

/**
 * TODO
 */
class SqlValueLoader extends NCValueLoader with LazyLogging {
    override def load(e: NCElement): java.util.Set[NCValue] = {
        if (!e.getGroups.contains("column"))
            throw new IllegalArgumentException(s"Unexpected element: ${e.getId}")

        val tab: String = e.meta("sql:tablename")
        val col: String = e.meta("sql:name")

        if (tab == null || col == null)
            throw new IllegalArgumentException(s"Missed required metadata for element: ${e.getId}")

        SqlAccess.select(new NCSqlQuery {
            override def getSql: String = s"SELECT $col FROM $tab WHERE $col IS NOT NULL"
            override def getParameters: util.List[AnyRef] = Seq.empty.asJava
        }, false).
            rows.
            map(_.head).
            map(_.toString.trim).
            filter(!_.isEmpty).
            map(
                v ⇒ new NCValue {
                    override def getName: String = v
                    override def getSynonyms: util.List[String] = Collections.singletonList(v)
                    override def toString: String = s"Value: $v"
                }
            ).toSet.asJava
    }
}
